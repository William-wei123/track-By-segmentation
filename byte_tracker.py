import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
import lib.utils.snake.snake_poly_utils as Sn_poly


from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState

# for REID
from .nn_matching import NearestNeighborDistanceMetric
from ..deep.feature_extractor import Extractor
from ..tool import *
import cv2
class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score ,poly=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.observe = None
        self.observe_his = None
        self.score = score
        self.tracklet_len = 0
        self.tracklet_len_ture = 0

        self.trust_rate = 1
        self.blocked_num = 0

        self.Poly = poly
        self.Poly_his = poly
        self.UP = 0
        self.DOWN = 0
        self.LEFT = 0
        self.RIGHT = 0



    def predict(self):
        mean_state = self.mean.copy()
        trust_rate = self.trust_rate#self.trust_rate
        if self.tracklet_len < 10:
            trust_rate = trust_rate / 4
        elif self.tracklet_len < 15:
            trust_rate = trust_rate / 3
        elif self.tracklet_len < 20:
            trust_rate = trust_rate / 2
        if self.state != TrackState.Tracked:

            mean_state[7] = 0   ##保持宽度增长不变
            mean_state[4:7] = mean_state[4:7]*trust_rate   #保持
            mean_state[5] = mean_state[5]/3   #保持竖直方向的速度小
        else:
            mean_state[7:] = 0  ##保持宽度增长不变
            mean_state[4:6] = mean_state[4:6] * trust_rate  # 保持
            mean_state[5] = mean_state[5]/3  # 保持竖直方向的速度小  / 2
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)


    def predict_his(self):
        mean_state = self.history.copy()
        if self.state != TrackState.Tracked:
            trust_rate = self.trust_rate
            if self.tracklet_len<10:
                trust_rate = trust_rate/4
            elif self.tracklet_len<15:
                trust_rate = trust_rate / 3
            elif self.tracklet_len < 20:
                trust_rate = trust_rate / 2
            mean_state[7] = 0   ##保持宽度增长不变
            mean_state[4:7] = mean_state[4:7]*trust_rate   #保持
            mean_state[5] = mean_state[5]/2     #保持竖直方向的速度小
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:      #表示lost态就不预测了？ cwp  记得改回来
                    multi_mean[i][7] = 0    #保持宽度增长不变
                    # multi_mean[i][6] = 0  # 保持高宽比增长不变
                    # multi_mean[i][5] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.history = self.mean

        self.tracklet_len = 0
        self.tracklet_len_ture = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True

        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.observe = self.tlwh
        self.observe_his = self.observe

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.tlwh_to_xyah(new_track.tlwh))
        # self.tracklet_len = 0 #为了能解决掉帧的问题，先暂时不要重新赋值
        self.blocked_state = False
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id    #被记为lost态的frame_id等于end_frame
        new_tlwh = new_track.tlwh
        if self.state is TrackState.Lost:
            if self.blocked_state is True:   #被判定为阻挡
                self.trust_rate = 0.95      #充分相信卡尔曼滤波
                self.tracklet_len += 1
                if not self.Poly is False:
                    self.Poly_his = self.Poly
                    new_track.Poly = new_track.Poly + torch.Tensor(new_track.mean[:2] - new_track.history[:2])
                    self.Poly = new_track.Poly#+new_track.mean[:2]-new_track.history[:2]
                    pass
            # else:
            if self.trust_rate >0.05:
                self.trust_rate-= 0.05
            pass
            # pass
        else:   #tracked
            self.tracklet_len += 1
            self.state = TrackState.Tracked
            self.observe_his = self.observe
            self.observe = new_track.tlwh
            if self.blocked_state is True:   #被判定为阻挡
                temp_tlbr = self.tlwh2tlbr(self.observe)
                this_tlbr = self.tlwh2tlbr(self.tlwh)
                if self.DOWN>0 and self.RIGHT>0:#方向为右下  匹配左上角坐标
                    xyerroe = this_tlbr[0:2] - temp_tlbr[0:2]#self.tlwh2tlbr(self.observe)
                    self.mean[0:2] = self.mean[0:2] - xyerroe
                elif self.DOWN<0 and self.RIGHT<0:#方向为左上  匹配右下角坐标:
                    xyerroe = this_tlbr[2:] - temp_tlbr[2:]  # self.tlwh2tlbr(self.observe)
                    self.mean[0:2] = self.mean[0:2] - xyerroe

                self.predict()
                new_tlwh = self.tlwh
            self.tracklet_len_ture += 1
        if self.mean[4]>0:
            self.DOWN +=1
            self.UP -= 1
        else:
            self.DOWN -= 1
            self.UP += 1
        if self.mean[5] > 0:
            self.RIGHT += 1
            self.LEFT -= 1
        else:
            self.RIGHT -= 1
            self.LEFT += 1

        #add history mean
        self.history = self.mean

        # new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh),
                                                               )

        self.is_activated = True

        self.score = new_track.score

        if not self.Poly is False:
            self.Poly_his = self.Poly
            self.Poly = new_track.Poly
            pass

    def update_his(self):
    # add history mean
        if self.blocked_state is True:
            if self.tracklet_len>20:
                self.predict_his()
            else:
                self.mean[2:5] = self.history[2:5]
                self.mean[6:] = 0

                self.history = self.mean
        # else:
        self.history = self.mean

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def get_history2tlbr(self):
        """
        把历史信息转化成trbl(min x, min y, max x, max y)格式，便于IOU计算是否阻挡
        """
        if self.history is None:
            return self._tlwh.copy()
        ret = self.history[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        ret[2:] += ret[:2]
        return ret
    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def tlwh2tlbr(self,tlwh):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret


    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)



class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh/10       #+0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # model_path = '/home/xinqiang_329/桌面/cwp/mmdetection/mmlab_test/track/deepsort/deep/checkpoint/ckpt_ship1.t7'
        # self.extractor = Extractor(model_path, use_cuda=True)

    # def _get_features(self, bbox_xywh, ori_img):    #for REID
    #     im_crops = []
    #     num = 0
    #     for box in bbox_xywh:
    #         x1,y1,x2,y2 = box   #self._xywh_to_xyxy(box)  #本来就是xyxy 格式
    #         if y2-y1<1 or x2-x1<1:  #太小的框会报错0
    #             continue
    #         im = ori_img[int(y1):int(y2), int(x1):int(x2)]
    #         # cv2.imshow('11111', im)
    #         # cv2.waitKey(50)
    #         print(box)
    #         im_crops.append(im)
    #         # cv2.imwrite('%d.jpg'%num,im)#用于保存探测框里面的图，用来做一些测试
    #         # num+=1
    #     if im_crops:
    #         features = self.extractor(im_crops)
    #     else:
    #         features = np.array([])
    #     return features

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def judge_blockforstracks(self, stracksa, stracksb):
        '''
        一般判断stracksa 里面是不是有被stracksb阻挡
        blockeds为（a,b）stracksa中索引a被stracksb中索引b阻挡
        u_blockeds 是stracksa中未被阻挡的索引
        '''
        dists = matching.iou_distance(stracksa, stracksb)  # 那么他为什么就不会是重新赋予ID呢？

        blockeds, u_blockeds, _ = matching.linear_assignment(dists, thresh=0.999)  # 得出被阻挡的索引

        return blockeds, u_blockeds
    def judge_blockfortlbr(self, tlbra, tlbrb):
        '''
        一般判断stracksa 里面是不是有被stracksb阻挡
        blockeds为（a,b）stracksa中索引a被stracksb中索引b阻挡
        u_blockeds 是stracksa中未被阻挡的索引
        '''
        _iou = matching.ious(tlbra, tlbrb)  # 那么他为什么就不会是重新赋予ID呢？
        dists = 1-_iou
        blockeds, u_blockeds, _ = matching.linear_assignment(dists, thresh=0.999)  # 得出被阻挡的索引

        return blockeds, u_blockeds
    def fromstrck2poly_iou_dist(self, atracks,btracks):
        """
        Compute cost based on IoU
        :type atracks: list[STrack]
        :type btracks: list[STrack]

        :rtype cost_matrix np.ndarray
        """

        if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
                len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
            atlbrs = atracks
            btlbrs = btracks
        else:
            aPolys = [track.Poly for track in atracks]
            btlbrs = [track.Poly for track in btracks]
        _ious = Sn_poly.get_poly_iou_matrix(aPolys, btlbrs)
        cost_matrix = 1 - _ious

        return cost_matrix
    def update(self, output_results, img_info, img_size, ori_imgh=None , Ploy=None):    #ori_imgh for REID  #加poly前
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale


        remain_inds = scores > self.args.track_thresh
        inds_low = scores > self.args.track_thresh_low
        inds_high = scores < self.args.track_thresh

        # \/important
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        # /\important

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)   #指上一帧刚被识别到的
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''#lost_stracks为曾经标记过，但丢失了一帧或几帧
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)

        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''  #因为阈值调低，几乎没用到此不部分
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]#第一次分高每匹配上的track  给r_...
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.8)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
                # 先解决掉帧的问题

                #判断是否被遮挡-->

                #然后根据卡尔曼预测当前帧位置-->

        # 试图加入REID--------->    cwp
        if ori_imgh is None:
            pass
        else:
            # pass
            # 借鉴deepsort代码-------->
            # 基于外观信息和马氏距离，计算卡尔曼滤波预测的tracks和当前时刻检测到的detections的代价矩阵
            def gated_metric(tracks, dets, track_indices, detection_indices):  # metric 自己加入的
                features = np.array([dets[i].feature for i in detection_indices])
                targets = np.array([tracks[i].track_id for i in track_indices])

                # 基于外观信息，计算tracks和detections的余弦距离代价矩阵
                cost_matrix = self.metric.distance(features, targets)

                # 基于马氏距离，过滤掉代价矩阵中一些不合适的项 (将其设置为一个较大的值)
                # cost_matrix = linear_assignment.gate_cost_matrix(
                #     self.kf, cost_matrix, tracks, dets, track_indices,
                #     detection_indices)

                return cost_matrix


            # <------------------借鉴deepsort代码
            self.width = img_info[1]    #获取高宽
            self.height = img_info[0]
            # features = self._get_features(dets, ori_imgh)#bboxes可能有问题，需要识别一下是否满足输入格式
            #获取需要提取的特征，做匹配
            max_dist = 0.2
            max_cosine_distance = max_dist
            nn_budget = 100
            self.metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
            # cost_matrix = gated_metric()#需要得到需要计算的探测框索引和此时还没匹配上的轨迹ID
            # matches, u_unconfirmed, u_detection = matching.linear_assignment(cost_matrix, thresh=0.5)
        # <---------试图加入REID    cwp

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.9)#0.7
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:         #操作空间，大于多少帧他就抛弃掉
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)    #???sub_stracks??   去掉a中是b子集的元素，返回a中不是b子集的元素
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)#去掉重复
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        #######################0621#############################
            #######################0625#############################
        # 判断tracked态是否被阻挡--->效果一般
        for i,track in enumerate(output_stracks):

            other_track_observe = [tempt.tlwh2tlbr(tempt.observe) for tempi, tempt in enumerate(output_stracks) if tempi != i]
            blockeds, u_blockeds = self.judge_blockfortlbr([track.tlwh2tlbr(track.observe_his)], other_track_observe)
            # print(blockeds)
            if len(blockeds) > 0:
                h_e = track.observe_his[3] - track.observe[3]
                w_e = track.observe_his[2] - track.observe[2]
                area_e = track.observe_his[3]*track.observe_his[2] - track.observe[3]*track.observe[2]
                if area_e>0:  # 高度或者宽度在减小比较大，认为有可能被阻挡

                    track.blocked_state = True
                    if track.blocked_num < 11:
                        track.blocked_num += 1
                else:
                    if track.blocked_num > 0:
                        track.blocked_num -= 1
                    if track.blocked_num < 5:
                        track.blocked_state = False
            else:
                if track.blocked_num > 0:
                    track.blocked_num -= 1
                if track.blocked_state == True:
                    h_e = track.observe_his[3] - track.observe[3]
                    w_e = track.observe_his[2] - track.observe[2]
                    area_e = track.observe_his[3] * track.observe_his[2] - track.observe[3] * track.observe[2]
                    # if area_e > 0:  # 高度或者宽度在减小比较大，认为有可能被阻挡  or track.blocked_num > 5
                    #
                    #     track.blocked_state = True
                    #     if track.blocked_num < 11:
                    #         track.blocked_num += 1

                    if track.blocked_num < 5:
                        track.blocked_state = False

            # track.update_his()
        # <---判断tracked态是否被阻挡

        # output_stracks_history_trbl = [track.get_history2tlbr for track in output_stracks]
        # 判断lost_stracks是否被阻挡
        # 那么他为什么就不会是重新赋予ID呢？

        blockeds, u_blockeds = self.judge_blockforstracks(self.lost_stracks, output_stracks)    #得出被阻挡的索引

        for blocked, idet in blockeds:
            if self.lost_stracks[blocked].blocked_state is False:
                self.lost_stracks[blocked].blocked_state = True
                track.blocked_num += 1

            # else:
            #     pass
            self.lost_stracks[blocked].trust_rate = 0.9
        for u_blocked in u_blockeds:
            if self.lost_stracks[u_blocked] is True:
                # track.mark_removed()    #被阻挡后，恢复没被阻挡的情况，是不是应该被识别到，如果依然没被识别到，是不是可以把他删掉了？？
                pass
            self.lost_stracks[u_blocked].blocked_state = False
            # det = detections[idet]
            # if track.state == TrackState.Tracked:
            #     track.update(detections[idet], self.frame_id)
            #     activated_starcks.append(track)
            # else:
            #     track.re_activate(det, self.frame_id, new_id=False)
            #     refind_stracks.append(track)

        if self.frame_id == 229:
            print('****************')
        for track in self.lost_stracks:#画出掉帧的框
            if (track.tracklet_len < 4 or self.frame_id - track.end_frame > 5 or self.frame_id - track.tracklet_len>15) and track.blocked_state is False:#如果跟踪上的帧数<4,或者已经掉了5帧
                continue


            track.predict()
            track.update(track, track.end_frame)
            if track.tracklet_len_ture < 10:#W真正跟踪上少于10帧的lost预测后也不显示
                continue
            output_stracks.append(track)
        ########################0621###############################

        return output_stracks   #原先只画出跟踪上的且激活态的，也就是暂时不画出只检测出一帧的（第一帧除外）现在可以画出lost的状态

    def update_0626(self, output_results, img_info, img_size, ori_imgh=None , Ploy=None):    #ori_imgh for REID  加poly后
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale


        remain_inds = scores > self.args.track_thresh
        inds_low = scores > self.args.track_thresh_low
        inds_high = scores < self.args.track_thresh

        # \/important
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        if not Ploy is False:
            poly_keep = Ploy[remain_inds]
            poly_second = Ploy[inds_second]
        scores_second = scores[inds_second]
        # /\important

        if len(dets) > 0:
            '''Detections'''
            if not Ploy is False:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, poly) for
                              (tlbr, s, poly) in zip(dets, scores_keep, poly_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)   #指上一帧刚被识别到的
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''#lost_stracks为曾经标记过，但丢失了一帧或几帧
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        # if not Ploy is False:
        #     dists = self.fromstrck2poly_iou_dist(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''  #因为阈值调低，几乎没用到此不部分
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            if not Ploy is False:
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, poly) for
                              (tlbr, s, poly) in zip(dets_second, scores_second, poly_second)]
            else:
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if (strack_pool[i].state == TrackState.Tracked or strack_pool[i].state == TrackState.Lost)]#第一次分高每匹配上的track  给r_...



        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        # if not Ploy is False:
        #     dists = self.fromstrck2poly_iou_dist(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.9)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
                # 先解决掉帧的问题

                #判断是否被遮挡-->

                #然后根据卡尔曼预测当前帧位置-->

        # 试图加入REID--------->    cwp
        if ori_imgh is None:
            pass
        else:
            # pass
            # 借鉴deepsort代码-------->
            # 基于外观信息和马氏距离，计算卡尔曼滤波预测的tracks和当前时刻检测到的detections的代价矩阵
            def gated_metric(tracks, dets, track_indices, detection_indices):  # metric 自己加入的
                features = np.array([dets[i].feature for i in detection_indices])
                targets = np.array([tracks[i].track_id for i in track_indices])

                # 基于外观信息，计算tracks和detections的余弦距离代价矩阵
                cost_matrix = self.metric.distance(features, targets)

                # 基于马氏距离，过滤掉代价矩阵中一些不合适的项 (将其设置为一个较大的值)
                # cost_matrix = linear_assignment.gate_cost_matrix(
                #     self.kf, cost_matrix, tracks, dets, track_indices,
                #     detection_indices)

                return cost_matrix


            # <------------------借鉴deepsort代码
            self.width = img_info[1]    #获取高宽
            self.height = img_info[0]
            # features = self._get_features(dets, ori_imgh)#bboxes可能有问题，需要识别一下是否满足输入格式
            #获取需要提取的特征，做匹配
            max_dist = 0.2
            max_cosine_distance = max_dist
            nn_budget = 100
            self.metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
            # cost_matrix = gated_metric()#需要得到需要计算的探测框索引和此时还没匹配上的轨迹ID
            # matches, u_unconfirmed, u_detection = matching.linear_assignment(cost_matrix, thresh=0.5)
        # <---------试图加入REID    cwp

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)

        # if not Ploy is False:
        #     dists = self.fromstrck2poly_iou_dist(unconfirmed, detections)



        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.9)#0.7
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:         #操作空间，大于多少帧他就抛弃掉
            if self.frame_id - track.end_frame > self.max_time_lost and track.blocked_state is False:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)    #???sub_stracks??   去掉a中是b子集的元素，返回a中不是b子集的元素
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)#去掉重复
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        #######################0621#############################
            #######################0625#############################
        # 判断tracked态是否被阻挡--->效果一般
        for i,track in enumerate(output_stracks):

            other_track_observe = [tempt.Poly for tempi, tempt in enumerate(output_stracks) if tempi != i]
            # blockeds, u_blockeds = self.judge_blockfortlbr([track.tlwh2tlbr(track.observe_his)], other_track_observe)

            blockeds, u_blockeds = Sn_poly.get_poly_match_ind([track.Poly_his], other_track_observe)
            # print(blockeds)
            if len(blockeds) > 0:
                h_e = track.observe_his[3] - track.observe[3]
                w_e = track.observe_his[2] - track.observe[2]
                area_e = track.observe_his[3]*track.observe_his[2] - track.observe[3]*track.observe[2]
                if area_e>0:  # 高度或者宽度在减小比较大，认为有可能被阻挡

                    track.blocked_state = True
                    if track.blocked_num < 11:
                        track.blocked_num += 1
                else:
                    if track.blocked_num > 0:
                        track.blocked_num -= 1
                    if track.blocked_num < 1:
                        track.blocked_state = False
            else:
                if track.blocked_num > 0:
                    track.blocked_num -= 1
                if track.blocked_state == True:
                    h_e = track.observe_his[3] - track.observe[3]
                    w_e = track.observe_his[2] - track.observe[2]
                    area_e = track.observe_his[3] * track.observe_his[2] - track.observe[3] * track.observe[2]
                    if area_e > 0:  # 高度或者宽度在减小比较大，认为有可能被阻挡  or track.blocked_num > 5

                        track.blocked_state = True
                        if track.blocked_num < 11:
                            track.blocked_num += 1

                    if track.blocked_num < 1:
                        track.blocked_state = False

            # track.update_his()
        # <---判断tracked态是否被阻挡

        # output_stracks_history_trbl = [track.get_history2tlbr for track in output_stracks]
        # 判断lost_stracks是否被阻挡
        # 那么他为什么就不会是重新赋予ID呢？

        blockeds, u_blockeds = self.judge_blockforstracks(self.lost_stracks, output_stracks)    #得出被阻挡的索引

        for blocked, idet in blockeds:
            if self.lost_stracks[blocked].blocked_state is False:
                self.lost_stracks[blocked].blocked_state = True
                track.blocked_num += 1

            # else:
            #     pass
            self.lost_stracks[blocked].trust_rate = 0.99
        for u_blocked in u_blockeds:
            if self.lost_stracks[u_blocked] is True:
                # track.mark_removed()    #被阻挡后，恢复没被阻挡的情况，是不是应该被识别到，如果依然没被识别到，是不是可以把他删掉了？？
                pass
            self.lost_stracks[u_blocked].blocked_state = False
            # det = detections[idet]
            # if track.state == TrackState.Tracked:
            #     track.update(detections[idet], self.frame_id)
            #     activated_starcks.append(track)
            # else:
            #     track.re_activate(det, self.frame_id, new_id=False)
            #     refind_stracks.append(track)

        if self.frame_id == 229:
            print('****************')

        for track in self.lost_stracks:#画出掉帧的框
            # if (track.tracklet_len < 4 or self.frame_id - track.end_frame > 5 or track.end_frame - track.tracklet_len>15) and track.blocked_state is False:#如果跟踪上的帧数<4,或者已经掉了5帧
            #     continue


            track.predict()
            track.update(track, track.end_frame)
            if track.tracklet_len_ture < 10:#W真正跟踪上少于10帧的lost预测后也不显示
                continue
            if track.mean[3]/track.mean[2]<20 and (track.mean[0]<0 or track.mean[0]>img_w):
                continue
            '''
            output_stracks.append(track)
            '''
        ########################0621###############################

        return output_stracks   #原先只画出跟踪上的且激活态的，也就是暂时不画出只检测出一帧的（第一帧除外）现在可以画出lost的状态


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
