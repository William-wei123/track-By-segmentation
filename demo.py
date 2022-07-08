# -*- codeing = utf-8 -*-
import numpy
import torch.utils.data as data
import glob
import os
import cv2
import numpy as np
from lib.utils.snake import snake_config
# from lib.utils import data_utils
from lib.config import cfg
from lib.utils import img_utils, data_utils
import tqdm
import torch
from lib.networks import make_network
from lib.utils.net_utils import load_network
from lib.visualizers import make_visualizer
from tools.My_tool import SET_COMPOSITOR
# from cython_bbox import bbox_overlaps as bbox_ious
from lib.utils.data_utils import box_iou        #for IOU
from lib.csrc.extreme_utils import _ext         #for NMS

import PIL
class Dataset(data.Dataset):        #继承Dataset，
    def __init__(self):             #改写__init__方法
        super(Dataset, self).__init__()         #子类构造函数调用super().init()

        if os.path.isdir(cfg.demo_path):
            self.imgs = glob.glob(os.path.join(cfg.demo_path, '*'))
        elif os.path.exists(cfg.demo_path):
            self.imgs = [cfg.demo_path]
        else:
            raise Exception('NO SUCH FILE')

    def normalize_image(self, inp):
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - snake_config.mean) / snake_config.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def __getitem__(self, index):
        img = self.imgs[index]
        img = cv2.imread(img)

        width, height = img.shape[1], img.shape[0]
        center = np.array([width // 2, height // 2])
        scale = np.array([width, height])
        x = 32
        input_w = (int(width / 1.) | (x - 1)) + 1
        input_h = (int(height / 1.) | (x - 1)) + 1

        trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

        inp = self.normalize_image(inp)
        ret = {'inp': inp}
        meta = {'center': center, 'scale': scale, 'test': '', 'ann': ''}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.imgs)



def demo():

    network = make_network(cfg).cuda()              #看懂这句话成功一半                                                     #类    cuda()将操作对象放在GPU内存中,加速运算？
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)       #data/model/rcnn_snake/long_rcnn              #加载神经网络模型？具体参数是什么意思？
    network.eval()                                  #下载好别人训练好的模型而不需要自己训练时，运行程序就应该变为eval()模式。                                                                        #eval() 函数用来执行一个字符串表达式，并返回表达式的值。??

    dataset = Dataset()                 #加载数据集,提取照片 是不是只加载文件目录（图片名）
    visualizer = make_visualizer(cfg)
    # print(type(dataset))
    # print(type(visualizer))
    # print(dataset.imgs[0])
    dataset.imgs.sort(key=SET_COMPOSITOR)                 #函数 sort()用于列表中元素的排序
    # print(dataset.imgs[0])
    i = 0
    Sava_dir = os.path.split(os.path.split(cfg.demo_path)[0])[1]
    print(Sava_dir)
    for batch in tqdm.tqdm(dataset):        #tqdm显示进度条的库      .tqdm    分批处理
        _,Save_filename = os.path.split(dataset.imgs[i])
        print(f"i={i}")
        print(Save_filename)
        Save_filename ,_ =os.path.splitext(Save_filename)
        print(Save_filename)
        batch['inp'] = torch.FloatTensor(batch['inp'])[None].cuda()             #转换数据类型张量便于求导，为了传参给network
        with torch.no_grad():                                                   #被with torch.no_grad()包住的代码，不用跟踪反向梯度计算,减少内存?
            output = network(batch['inp'], batch)

        if cfg.task == 'ct_rcnn':
            print(output)
            # return 0
        else:
            print(output.keys())
            # return 0
        if cfg.task == 'rcnn_snake' or cfg.task == 'snake' or cfg.task == 'ct_rcnn':            #ct_rcnn   是自己的，有bug,好像训练只有识别
            if not os.path.exists('/home/xinqiang_329/desk/out_put/snake/'+Sava_dir+'/'+cfg.model):
                os.makedirs('/home/xinqiang_329/desk/out_put/snake/'+Sava_dir+'/'+cfg.model)
            Save_filename = os.path.join(Sava_dir+'/'+cfg.model,Save_filename)
            visualizer.visualize(output, batch,Save_filename)                                     #可视化部分，保存图像可以在里面加，但我想把它独立出来？
        else:
            visualizer.visualize(output, batch)
        print(list(output))               #这玩意是个字典,list获取字典关键字列表
        print(list(batch))          #这玩意也是个字典
        #cv2.imwrite('./demo_images/MyData/Output'+str(i)+'.png',np.ndarray(batch['inp']))
        i += 1

def draw_result(img,bbox=None, color=0, py=None, blocked_stated = None,mask=True,line=None,frame_idx=None):#for all
    if type(img) is str:
        image = cv2.imread(img)
    else:
        image = img

    color_list = [(0, 0, 1), (0, 1, 0), (1, 0, 0),(1, 1, 0)]
    color = color_list[color]
    if bbox is None:
        pass
    else:
        if type(bbox) is np.ndarray:
            reslut = bbox
        elif type(bbox) is torch.Tensor:
            reslut = bbox.cpu().numpy()
        for ind, detection in enumerate(reslut):
            x_min, y_min, x_max, y_max = detection[:4].astype(int)
            # figture.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], color='w', linewidth=0.5)
            Id = (detection[4])

            if type(Id) is np.int64:
                Id = Id.astype(np.uint8)
                cv2.putText(image, 'ID:'+str(Id), (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)  # ID
                # cv2.putText(image, 'track_num:' + str(reslut.shape[0]), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 2,
                #             color)  # track_num 跟踪到的数量
            else:
                Id = (Id).astype(np.uint8)  #*100暂时不画出分数了
                cv2.putText(image, 'ID:'+str(Id), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)  # 分数
                # cv2.putText(image, 'det_num:' + str(reslut.shape[0]), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 2,
                #             color)  # det_num det到的数量
            # figture.text((x_min+x_max)/2,(y_min+y_max)/2, str(Id))
            # print(detection)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)#探测框
            if blocked_stated != None:
                cv2.putText(image, blocked_stated[ind], (x_max, y_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color)  # det_num 跟踪到的数量

            # if save_mp4 == 1:
    if py is None:
        pass
    else:
        print(py.shape)
        # pts = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], np.int32)  # 数据类型必须为 int32
        # pts = pts.reshape((-1, 1, 2))
        # print(pts.shape)
        image = cv2.polylines(image, np.int32(py), color=color, isClosed=True)#shift=3#轮廓
        if mask is True:
            image_mask = np.zeros_like(image)
            # alpha 第一张图的透明度
            alpha = 1
            # beta 第2张图的透明度
            beta = 0.4
            gamma = 0
            temp_np = np.int32(py)#.cpu().numpy()
            image_mask = cv2.fillPoly(image_mask, temp_np, color=(1, 0, 0))
            cv2.imshow('mask', image_mask)
            cv2.waitKey(5)
            image = cv2.addWeighted(image,alpha, image_mask, beta, gamma)

    if line is None:
        pass
    else:
        # 以下为画轨迹，原理就是将前后帧同ID的跟踪框中心坐标连接起来
        if frame_idx + 1 > 2:
            for key, value in line.items():
                for a in range(len(value) - 1):
                    # color = COLORS_10[key % len(COLORS_10)]
                    index_start = a
                    index_end = index_start + 1
                    cv2.line(image, tuple(map(int, value[index_start])), tuple(map(int, value[index_end])),
                             # map(int,"1234")转换为list[1,2,3,4]
                             (255, 0, 0), thickness=2, lineType=8)

    ##########################################################################
    return image


# for deepsort
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

# for ByteTrack
def _tlwh_to_xyxy( bbox_tlwh, width, height):
    """
    TODO:
        Convert bbox from xtl_ytl_w_h to xc_yc_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    x,y,w,h = bbox_tlwh
    x1 = max(int(x),0)
    x2 = min(int(x+w), width-1)
    y1 = max(int(y),0)
    y2 = min(int(y+h), height-1)
    return [x1,y1,x2,y2]



def test_demo():#cwp
    save_mp4 = 0

    network = make_network(cfg).cuda()              #看懂这句话成功一半                                                     #类    cuda()将操作对象放在GPU内存中,加速运算？
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)       #data/model/rcnn_snake/long_rcnn              #加载神经网络模型？具体参数是什么意思？
    network.eval()                                  #下载好别人训练好的模型而不需要自己训练时，运行程序就应该变为eval()模式。                                                                        #eval() 函数用来执行一个字符串表达式，并返回表达式的值。??

    dataset = Dataset()                 #加载数据集,提取照片 是不是只加载文件目录（图片名）
    # visualizer = make_visualizer(cfg)
    # print(type(dataset))
    # print(type(visualizer))
    # print(dataset.imgs[0])
    dataset.imgs.sort(key=SET_COMPOSITOR)                 #函数 sort()用于列表中元素的排序
    # print(dataset.imgs[0])
    i = 0
    Sava_dir = os.path.split(os.path.split(cfg.demo_path)[0])[1]
    print(Sava_dir)
    second_model = 'yolox'
    csvdir = '/home/xinqiang_329/桌面/cwp/snake/out_put/csvfile/%s/%s'%(second_model, Sava_dir)#获取CSV的路径
    for batch in tqdm.tqdm(dataset):        #tqdm显示进度条的库      .tqdm    分批处理

        _,Save_filename = os.path.split(dataset.imgs[i])


        print(f"i={i}")
        print(Save_filename)
        Save_filename ,_ =os.path.splitext(Save_filename)
        print(Save_filename)
        temp_img = cv2.imread(dataset.imgs[i])


        if i==0 and save_mp4 == 1:
            fps = 4
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_dir = 'out_put/%s_yolox_%s.MP4' % (Save_filename,cfg.model)  # 因为最后有/，所以多来一次
            videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, (batch['inp'].shape[2], batch['inp'].shape[1]), True)  # 输出路径和文件名，输出格式？，帧率，图片大小（不放心可以读一张，根据图片自动设置）

        ###############################
        csv_filename = '%s/%s.csv' % (csvdir, Save_filename)
        second_m_data = np.loadtxt(csv_filename, delimiter=',', dtype=float)#读取对应图像其他检测算法的检测结果
        #计算输出结果和原图的比值
        A = batch['inp'].shape[2]/batch['meta']['scale'][0]
        B = batch['inp'].shape[1]/batch['meta']['scale'][1]
        second_m_data[:, 0] = second_m_data[:, 0]*A
        second_m_data[:, 2] = second_m_data[:, 2] * A
        second_m_data[:, 1] = second_m_data[:, 1] * B
        second_m_data[:, 3] = second_m_data[:, 3] * B
        second_m_data[:, 0:4] = second_m_data[:, 0:4] / snake_config.down_ratio#为了匹配deepsnake的输出
        class_second_m_data = np.zeros((second_m_data.shape[0], 1))     #类别  0  表示船
        second_m_data = np.column_stack((second_m_data, class_second_m_data))#加入到其他检测算法的结果中
        second_m_data_ct = np.column_stack((second_m_data[:, 0] + second_m_data[:, 2], second_m_data[:, 1] + second_m_data[:, 3]))
        second_m_data_ct = second_m_data_ct/2   #获取中心位置
        output_second = {}  #用于gcn回归轮廓
        output_second['detection'] = torch.FloatTensor(second_m_data).cuda().unsqueeze(0)
        output_second['ct'] = torch.IntTensor(second_m_data_ct).cuda().unsqueeze(0)
        batch['inp'] = torch.FloatTensor(batch['inp'])[None].cuda()             #转换数据类型张量便于求导，为了传参给network

        with torch.no_grad():                                                   #被with torch.no_grad()包住的代码，不用跟踪反向梯度计算,减少内存?
            [output, cnn_feature]= network.forward_1(batch['inp'], batch)#output=network(batch['inp'], batch)
        # output['detection'] = torch.FloatTensor(second_m_data).cuda()
        output_second['ct_hm'] = output['ct_hm']
        output_second['wh'] = output['wh']
        with torch.no_grad():      # 人为分解成两步，第二步                                            #被with torch.no_grad()包住的代码，不用跟踪反向梯度计算,减少内存?
            output= network.forward_2(output, cnn_feature, batch)
            output_second = network.forward_2(output_second, cnn_feature, batch)
        # second_m_data[:, 0:4] = second_m_data[:, 0:4]*snake_config.down_ratio
        # second_m_data = second_m_data[second_m_data[..., 4] > snake_config.ct_score]

        # image_result = draw_result(img_data, second_m_data, color=0)
        image_result = img_utils.bgr_to_rgb(
            img_utils.unnormalize_img(batch['inp'][0], snake_config.mean, snake_config.std).permute(1, 2,
                                                                                                    0)).detach().cpu().numpy()

        image_result = cv2.cvtColor(image_result, cv2.COLOR_RGB2BGR)  # 作者用的plt画图，plt格式为RGB，需要转为CV2的BGR
        box = output['detection'][:, :5]     #获取探测框
        box[:, :4] = box[:, :4] * snake_config.down_ratio
        bbox_deepsnake = box[:, :4]
        # box = box.detach().cpu().numpy()
        ex = output['py']                   #获取轮廓信息
        ex = ex[-1] if isinstance(ex, list) else ex##获取轮廓信息
        ploy = ex * snake_config.down_ratio  #restoy    #.detach().cpu().numpy()
        # image_result = draw_result(image_result, box, color=0)   #deepsnake  原始输出, py=ploy

        box = output_second['detection'][:, :5]
        box[:, :4] = box[:, :4] * snake_config.down_ratio   #restoy
        bbox_second = box[:, :4]
        # box = box.detach().cpu().numpy()
        ex = output_second['py']
        ex = ex[-1] if isinstance(ex, list) else ex
        ploy = ex * snake_config.down_ratio  # * snake_config.down_ratio    .detach().cpu().numpy()
        # image_result = draw_result(image_result, box, color=1)#yolox替换后的输出  , py=ploy
        # continue
        IOU = box_iou(bbox_second,bbox_deepsnake)
        if IOU.shape[1]==0: #deepsnake 一个船都没检测出来
            first_ind = (box[:, 4] > 0.1)
            pass
        else:
            print('第一次IOU匹配:', IOU.shape)
            IOU_score = 0.56     #IOU阈值设为0.5，阈值大于0.5的保存，小于0.5 的接下来看分数
            Iou_Value , IOU_ind = torch.max(IOU,1)
            first_ind = Iou_Value > IOU_score
        image_result = draw_result(image_result,color=2, py=ploy[first_ind])  # 画出第一次IOU匹配的结果  为了能够高级索引，所以先传cuda的tensor进来，后期需要转成cpu里面的numpy
        second_bbox_second = box[first_ind==False, :]
        second_py_second = ploy[first_ind==False, :]
        second_ind = (second_bbox_second[:, 4]>0.05)        ##二次阈值的地方
        second_bbox_second = second_bbox_second[second_ind]
        second_py_second = second_py_second[second_ind]
        # image_result = draw_result(image_result, second_bbox_second, color=2)  # 画出第二次IOU匹配的结果，和拉高阈值处理的分数
        # 进行第三次匹配，IOU匹配，为了去除重复检测，将第一次IOU 和第二分数匹配的结果再进行IOU ，如果有重复，（默认保留第一次IOU结果），第二次分数匹配去除
        IOU = box_iou(box[first_ind, :4], second_bbox_second[:, :4])
        if IOU.shape[1]==0 or IOU.shape[0]==0: #second_bbox_second 一个船都没
            # first_ind = (box[:, 4] > 0.3) #啥事不做
            pass
        else:
            print('第2次IOU匹配:{IOU.shape}',IOU.shape)
            IOU_score = 0.01     #IOU阈值设为0.5，阈值大于0.5的丢掉
            Iou_Value , IOU_ind = torch.max(IOU,0)  #改0，因为要丢掉第二次分数匹配的
            third = Iou_Value < IOU_score
            if third.sum() > 0:

                # 设想第四步nms解决现有船的重复识别问题？
                # 第五步实现跟踪，及会遇情况  见 test_demo_0618
                image_result = draw_result(image_result, bbox=second_bbox_second[third], color=0)  # 画出第3次匹配的结果，,py=second_py_second[third]
                pass
            else:
                pass
        if save_mp4 == 1:
            videoWriter.write((image_result*255).astype(np.uint8))
        else:
            cv2.imshow('result', image_result)
            cv2.waitKey(5)

        i += 1

# #######################
# 使用什么跟踪算法 sort_flag
# 0:自己瞎写 REID + Bytetrack
# 1:sort
# 2：deepsort
# 3:Bytetrack
# 4:tracktor?
# 5:QDtrack?
sort_flag = 3

#为了验证跟踪效果
def save_track_txt(txt_path, frame_idx, outputs):
    if  len(outputs) != 0:
        outputs_xywh = []
        for j, output in enumerate(outputs):
            bbox_left = output[0]
            bbox_top = output[1]
            bbox_w = output[2] - output[0]
            bbox_h = output[3] - output[1]
            identity = output[-1]
            with open(txt_path, 'a') as f:
                # f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                #                                bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                f.write("{},{},{},{},{},{},{},{},{},{}\n".format(frame_idx, identity, bbox_left,
                                                                 bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))
            outputs_xywh.append([int(bbox_left),int(bbox_top), int(bbox_w), int(bbox_h)])
        return outputs_xywh

def xyxy2tlwh(x):#forTrackLine
    '''
    (top left x, top left y,width, height)
    '''
    y = torch.zeros_like(x) if isinstance(x,
                                          torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y
    ##########################################################################
    # outputs = [x1, y1, x2, y2, track_id]
def fforTrackLine(outputs,dict_box:dict):
    if len(outputs) > 0:
        bbox_xyxy = outputs[:, :4]  # 提取前四列  坐标
        identities = outputs[:, -1]  # 提取最后一列 ID
        box_xywh = xyxy2tlwh(bbox_xyxy)
        # xyxy2tlwh是坐标格式转换，从x1, y1, x2, y2转为top left x ,top left y, w, h 具体函数看文章最后
        for j in range(len(box_xywh)):
            x_center = box_xywh[j][0] + box_xywh[j][2] / 2  # 求框的中心x坐标
            y_center = box_xywh[j][1] + box_xywh[j][3] / 2  # 求框的中心y坐标
            id = outputs[j][-1]
            center = [x_center, y_center]
            dict_box.setdefault(id, []).append(center)  # 这个字典需要提前定义 dict_box = dict()
    return 0

def test_demo_0618():#cwp  设想第四步nms解决现有船的重复识别问题？第五步实现跟踪，及会遇情况
    save_mp4 = 1
    save_json = True
    NMS_the = 0.4
    network = make_network(cfg).cuda()              #看懂这句话成功一半                                                     #类    cuda()将操作对象放在GPU内存中,加速运算？
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)       #data/model/rcnn_snake/long_rcnn              #加载神经网络模型？具体参数是什么意思？
    network.eval()                                  #下载好别人训练好的模型而不需要自己训练时，运行程序就应该变为eval()模式。                                                                        #eval() 函数用来执行一个字符串表达式，并返回表达式的值。??

    dataset = Dataset()                 #加载数据集,提取照片 是不是只加载文件目录（图片名）

    dataset.imgs.sort(key=SET_COMPOSITOR)                 #函数 sort()用于列表中元素的排序

    i = 0
    Sava_dir = os.path.split(os.path.split(cfg.demo_path)[0])[1]
    print(Sava_dir)
    second_model = 'yolox'
    csvdir = '/home/xinqiang_329/桌面/cwp/snake/out_put/csvfile/%s/%s'%(second_model, Sava_dir)#获取CSV的路径

    if sort_flag == 1:
        from track.sort import Sort
        mot_tracker = Sort()
    elif sort_flag == 2:
        from track.deepsort.deep_sort import DeepSort
        temp_cfg = {}
        temp_cfg[
            'REID_CKPT'] = "/home/xinqiang_329/桌面/cwp/mmdetection/mmlab_test/track/deepsort/deep/checkpoint/ckpt_ship1.t7"
        temp_cfg['MAX_DIST'] = 0.2  # 0.2  最大余弦距离
        temp_cfg['MIN_CONFIDENCE'] = 0.3  # 0.3  YOLOv5最小检测置信度，增大置信度可去除杂散干扰。
        temp_cfg['NMS_MAX_OVERLAP'] = 0.5  # 0.5
        temp_cfg['MAX_IOU_DISTANCE'] = 0.8  # 0.7 IOU最大距离，此值小则不易匹配，将产生新的ID。
        temp_cfg['MAX_AGE'] = 70  # 70
        temp_cfg['N_INIT'] = 5  # 3 track连续confirm数量，增大有助于减少新ID出现。
        temp_cfg['NN_BUDGET'] = 100  # track最大feature数量

        mot_tracker = DeepSort(temp_cfg['REID_CKPT'],
                               max_dist=temp_cfg['MAX_DIST'], min_confidence=temp_cfg['MIN_CONFIDENCE'],
                               nms_max_overlap=temp_cfg['NMS_MAX_OVERLAP'],
                               max_iou_distance=temp_cfg['MAX_IOU_DISTANCE'],
                               max_age=temp_cfg['MAX_AGE'], n_init=temp_cfg['N_INIT'], nn_budget=temp_cfg['NN_BUDGET'],
                               use_cuda=True)
    elif sort_flag == 3:
        from track.bytetrack.tracker.byte_tracker import BYTETracker
        import argparse
        import track.bytetrack_config as arg_track

        mot_tracker = BYTETracker(arg_track, frame_rate=5)
        print('*' * 9 + 'use Bytetrack' + '*' * 9)
        # ######################
    json_results = []   #用于保存结果作验资
    dict_box_line = {}
    for batch in tqdm.tqdm(dataset):        #tqdm显示进度条的库      .tqdm    分批处理

        _,Save_filename = os.path.split(dataset.imgs[i])
        # cv2.imshow('result', np.transpose(batch['inp'], (1, 2,0)))
        # cv2.waitKey(5)
        # continue
        print(f"i={i}")
        print(Save_filename)
        Save_filename ,_ =os.path.splitext(Save_filename)
        print(Save_filename)
        # temp_img = cv2.imread(dataset.imgs[i])

        if i==0 and save_mp4 == 1:
            fps = 4
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_dir = 'out_put/0708%s_yolox_%s_track0.9.MP4' % (Save_filename,cfg.model)  # 因为最后有/，所以多来一次
            videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, (batch['inp'].shape[2], batch['inp'].shape[1]), True)  # 输出路径和文件名，输出格式？，帧率，图片大小（不放心可以读一张，根据图片自动设置）

        ###############################
        csv_filename = '%s/%s.csv' % (csvdir, Save_filename)
        second_m_data = np.loadtxt(csv_filename, delimiter=',', dtype=float)#读取对应图像其他检测算法的检测结果
        #计算输出结果和原图的比值
        A = batch['inp'].shape[2]/batch['meta']['scale'][0]     #宽
        B = batch['inp'].shape[1]/batch['meta']['scale'][1]     #高
#nms-->
        second_m_data = torch.FloatTensor(second_m_data).cuda()
        box_ = second_m_data[:, 0:4]
        score_ = second_m_data[:, 4]
        ind = _ext.nms(box_, score_, NMS_the)   #0.5 为IOU阈值
        second_m_data = second_m_data[ind]
        second_m_data = second_m_data.cpu()
# <--nms
        second_m_data[:, 0] = second_m_data[:, 0]*A
        second_m_data[:, 2] = second_m_data[:, 2] * A
        second_m_data[:, 1] = second_m_data[:, 1] * B
        second_m_data[:, 3] = second_m_data[:, 3] * B
        second_m_data[:, 0:4] = second_m_data[:, 0:4] / snake_config.down_ratio#为了匹配deepsnake的输出
        class_second_m_data = np.zeros((second_m_data.shape[0], 1))     #类别  0  表示船
        second_m_data = np.column_stack((second_m_data, class_second_m_data))#加入到其他检测算法的结果中
        second_m_data_ct = np.column_stack((second_m_data[:, 0] + second_m_data[:, 2], second_m_data[:, 1] + second_m_data[:, 3]))
        second_m_data_ct = second_m_data_ct/2   #获取中心位置
        output_second = {}  #用于gcn回归轮廓
        output_second['detection'] = torch.FloatTensor(second_m_data).cuda().unsqueeze(0)
        output_second['ct'] = torch.IntTensor(second_m_data_ct).cuda().unsqueeze(0)
        batch['inp'] = torch.FloatTensor(batch['inp'])[None].cuda()             #转换数据类型张量便于求导，为了传参给network

        with torch.no_grad():                                                   #被with torch.no_grad()包住的代码，不用跟踪反向梯度计算,减少内存?
            [output, cnn_feature]= network.forward_1(batch['inp'], batch)#output=network(batch['inp'], batch)
        # output['detection'] = torch.FloatTensor(second_m_data).cuda()
        output_second['ct_hm'] = output['ct_hm']
        output_second['wh'] = output['wh']
        #在gcn之前把数据处理掉           0618
# 第一次IOU筛选，都检测出来的船 - -->
#         box = output['detection'][0, :, :5]     #获取探测框
#         bbox_deepsnake = box[:, :4]
#         box = output_second['detection'][0, :, :5]
#         bbox_second = box[:, :4]
#
#         IOU = box_iou(bbox_second, bbox_deepsnake)
#         if IOU.shape[1] == 0:  # deepsnake 一个船都没检测出来
#             first_ind = (box[:, 4] > 0.1)
#             pass
#         else:
#             print('第一次IOU匹配:', IOU.shape)
#             IOU_score = 0.7  # IOU阈值设为0.5，阈值大于0.5的保存，小于0.5 的接下来看分数
#             Iou_Value, IOU_ind = torch.max(IOU, 1)
#             first_ind = Iou_Value > IOU_score
#             print(first_ind.sum())
#         output_second['detection'] = output_second['detection'][:,first_ind,:]
#         output_second['ct'] = output_second['ct'][:,first_ind,:]
# <----第一次IOU筛选，都检测出来的船

# 第2次分数筛选， - -->
#
        with torch.no_grad():      # 人为分解成两步，第二步                                            #被with torch.no_grad()包住的代码，不用跟踪反向梯度计算,减少内存?
            output= network.forward_2(output, cnn_feature, batch)
            output_second = network.forward_2(output_second, cnn_feature, batch)
        # second_m_data[:, 0:4] = second_m_data[:, 0:4]*snake_config.down_ratio
        # second_m_data = second_m_data[second_m_data[..., 4] > snake_config.ct_score]

        # image_result = draw_result(img_data, second_m_data, color=0)
        image_result = img_utils.bgr_to_rgb(
            img_utils.unnormalize_img(batch['inp'][0], snake_config.mean, snake_config.std).permute(1, 2,
                                                                                                    0)).detach().cpu().numpy()
        image_result = cv2.cvtColor(image_result, cv2.COLOR_RGB2BGR)#作者用的plt画图，plt格式为RGB，需要转为CV2的BGR
        img0 = image_result     #for track

        box = output['detection'][:, :5]     #获取探测框
        box[:, :4] = box[:, :4] * snake_config.down_ratio
        bbox_deepsnake = box[:, :4]
        # box = box.detach().cpu().numpy()
        ex = output['py']                   #获取轮廓信息
        ex = ex[-1] if isinstance(ex, list) else ex##获取轮廓信息
        ploy = ex * snake_config.down_ratio  #restoy    #.detach().cpu().numpy()
        # image_result = draw_result(image_result, color=0, py=ploy)   #deepsnake  原始输出

        box = output_second['detection'][:, :5]
        box[:, :4] = box[:, :4] * snake_config.down_ratio   #restoy
        bbox_second = box[:, :4]
        # box = box.detach().cpu().numpy()
        ex = output_second['py']
        ex = ex[-1] if isinstance(ex, list) else ex
        ploy = ex * snake_config.down_ratio  # * snake_config.down_ratio    .detach().cpu().numpy()
        # image_result = draw_result(image_result, box, color=1)#yolox替换后的输出  , py=ploy
        # continue
        IOU = box_iou(bbox_second,bbox_deepsnake)
        if IOU.shape[1]==0: #deepsnake 一个船都没检测出来
            first_ind = (box[:, 4] > 0.1)
            pass
        else:
            print('第一次IOU匹配:', IOU.shape)
            IOU_score = 0.6     #IOU阈值设为0.5，阈值大于0.5的保存，小于0.5 的接下来看分数
            Iou_Value , IOU_ind = torch.max(IOU,1)
            first_ind = Iou_Value > IOU_score

        second_bbox_1_scc = box[first_ind, :]
        second_py_1_scc = ploy[first_ind]
        second_bbox_second = box[first_ind==False, :]
        second_py_second = ploy[first_ind==False, :]
        second_ind = (second_bbox_second[:, 4]>0.4)        ##1次阈值的地方 0.6 分数大于0.6，而且第一次IOU没有选出的继续保留
        second_bbox_1_scc = torch.cat((second_bbox_1_scc, second_bbox_second[second_ind]), 0)
        second_py_1_scc = torch.cat((second_py_1_scc, second_py_second[second_ind]), 0)

        #因为可以充分相信他们，所以把第一批分数底的认为提高
        second_bbox_1_scc[second_bbox_1_scc[:, 4]<0.3] = second_bbox_1_scc[second_bbox_1_scc[:, 4]<0.3]+0.1
        # image_result = draw_result(image_result, bbox=second_bbox_1_scc, color=0,
        #                            py=second_py_1_scc)  # 画出第一次IOU匹配and high score的结果  为了能够高级索引，所以先传cuda的tensor进来，后期需要转成cpu里面的numpy

        second_bbox_second = second_bbox_second[second_ind==False]
        second_py_second = second_py_second[second_ind==False]
        second_ind_2 = (second_bbox_second[:, 4]>0.01)        ##2次阈值的地方 0.  分数小于0.6而且第一次IOU没有被选上的
        second_bbox_second = second_bbox_second[second_ind_2]
        second_py_second = second_py_second[second_ind_2]
        # image_result = draw_result(image_result, second_bbox_second, color=2)  # 画出第二次IOU匹配的结果，和拉高阈值处理的分数
        ind = _ext.nms(second_bbox_second[:,:4], second_bbox_second[:,4], 0.1)  # 0.1 为IOU阈值  在分底里面只保存分最高的，尽管只有一点点接触
        second_bbox_second = second_bbox_second[ind]
        second_py_second = second_py_second[ind]
        # 进行第三次匹配，IOU匹配，为了去除重复检测，将第一次IOU 和第二分数匹配的结果再进行IOU ，如果有重复，（默认保留第一次IOU结果），第二次分数匹配去除
        IOU = box_iou(second_bbox_1_scc[:, :4], second_bbox_second[:, :4])
        if IOU.shape[1]==0 or IOU.shape[0]==0: #second_bbox_second 一个船都没
            # first_ind = (box[:, 4] > 0.3) #啥事不做
            ship_result = second_bbox_1_scc.cpu()
            # second_py_2_scc = second_py_second[third]
            py_result = second_py_1_scc.cpu()
            pass
        else:
            print('第2次IOU匹配:{IOU.shape}',IOU.shape)
            IOU_score = 0.01     #IOU阈值设为0.5，阈值大于0.5的丢掉
            Iou_Value , IOU_ind = torch.max(IOU,0)  #改0，因为要丢掉第二次分数匹配的
            third = Iou_Value < IOU_score
            if third.sum() > 0:

                # 设想第四步nms解决现有船的重复识别问题？
                # 第五步实现跟踪，及会遇情况  见 test_demo_0618
                # image_result = draw_result(image_result, bbox=second_bbox_second[third],py=second_py_second[third], color=1)  # 画出第3次匹配的结果，
                second_bbox_2_scc = second_bbox_second[third]
                ship_result = torch.cat((second_bbox_1_scc, second_bbox_2_scc), 0).cpu()
                second_py_2_scc = second_py_second[third]
                py_result = torch.cat((second_py_1_scc, second_py_2_scc), 0).cpu()
                pass
            else:
                # second_bbox_2_scc = second_bbox_second[third]
                ship_result = second_bbox_1_scc.cpu()
                # second_py_2_scc = second_py_second[third]
                py_result = second_py_1_scc.cpu()
                pass
#都需要传bbox
        # image_result = draw_result(image_result, bbox=ship_result, py=py_result,
        #                            color=1)  # 画出根据探测框结合最终的结果
        if sort_flag == 1:  # sort
            # temp_ind = ship_result[:, 4] > ct_score
            # print(temp_ind)
            # ship_result = ship_result[temp_ind, :]
            last_result = mot_tracker.update(ship_result)
        elif sort_flag == 2:  # deepsort
            bbox_xywh = xyxy2xywh(ship_result[:, :4])
            confs = ship_result[:, 4:5]
            # img0 = cv2.imread(img)

            last_result = mot_tracker.update(bbox_xywh, confs, img0)  # 图片中心坐标+宽高，置信度，BGR格式图片
        elif sort_flag == 3:  # bytetrack
            # img0 = cv2.imread(img)

            online_targets = mot_tracker.update_0626(ship_result, img0.shape, img0.shape,
                                                img0, Ploy=py_result)  # 左上角右下角+置信度，[img_info['height'], img_info['width']]，图片大小		两个图片大小好像是为了能恢复原图大小，可能原始代码会缩放
            # 返回的Strack类型需要特殊处理一下
            last_result = []
            blocked_state = []
            for t in online_targets:
                temp = []

                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                # if tlwh[2] * tlwh[3] > arg.min_box_area and not vertical:#暂时不懂这一部是为了什么，所以取消掉
                temp = _tlwh_to_xyxy(tlwh, img0.shape[1], img0.shape[0])
                temp.append(tid)
                blocked_statestr = ''
                if t.blocked_state is True:
                    blocked_statestr += 'b'

                if t.state is 2:
                    blocked_statestr += 'l'
                else:
                    blocked_statestr += 't'
                blocked_state.append(blocked_statestr)
                # online_tlwhs.append(tlwh)
                # online_ids.append(tid)
                # online_scores.append(t.score)
                last_result.append(temp)
            last_result = np.array(last_result)
        # image_result = draw_result(image_result, bbox=last_result, #py=second_py_second[third],
        #                            color=0,blocked_stated=blocked_state)  # 画出track的结果，
        last_result = torch.Tensor(last_result)

        FinalIOU = box_iou(ship_result[:,:4],last_result[:,:4])
        if FinalIOU.shape[1]==0 or FinalIOU.shape[0]==0: #deepsnake 一个船都没检测出来
            first_ind = (box[:, 4] > 0.1)
            Final_bbox_result = ship_result
            Final_py_result = py_result

        else:
            from track.bytetrack.tracker.matching import ious, linear_assignment  # for IOU
            print('Final IOU匹配:', FinalIOU.shape)
            FinalIOU = FinalIOU.numpy()
            FinalIOU = 1-FinalIOU
            matched, _, lost_tracked = linear_assignment(FinalIOU, 0.8)
            Final_bbox_result = []
            Final_py_result = []
            Final_bbox_resultforEev = []
            for ship_result_ind,last_result_ind in matched:
                temp = ship_result[ship_result_ind].numpy().copy()
                Final_bbox_result.append(temp)
                temp[4] = int(last_result[last_result_ind][4])
                Final_bbox_resultforEev.append(temp)
                Final_py_result.append(py_result[ship_result_ind].numpy())

            for lost_track in lost_tracked: #没被匹配上的track都是lost态的，将其结果添加到最终结果中，ID值什么的都在前面就已经画出
                Final_py_result.append(online_targets[lost_track].Poly_his.numpy())
                losttrack_temp = last_result[lost_track].numpy().copy()    #最终的bbox输出的是numpy类型，第5位是分数
                Final_bbox_resultforEev.append(losttrack_temp.copy())
                losttrack_temp[4] = 0.9#为了显示没画出的分数值
                Final_bbox_result.append(losttrack_temp)


            Final_bbox_result = np.array(Final_bbox_result) #探测框的真实值作为输出
            Final_py_result = np.array(Final_py_result)
            Final_bbox_resultforEev = np.array(Final_bbox_resultforEev)
        fforTrackLine(Final_bbox_resultforEev, dict_box_line)
        image_result = draw_result(image_result, bbox=Final_bbox_resultforEev,  py=Final_py_result,#Final_bbox_result
                                   color=1, blocked_stated = blocked_state,line=dict_box_line,frame_idx=i)  # 画出最终形态

        txt_path = 'out_put/oursMet_hua.txt'
        last_result = last_result.cpu().numpy()
        last_result = Final_bbox_resultforEev#Final_bbox_result#Final_bbox_resultforEev
        last_result[:,0] = last_result[:,0]/A
        last_result[:, 2] = last_result[:, 2] / A
        last_result[:, 1] = last_result[:, 1] / B
        last_result[:, 3] = last_result[:, 3] / B
        last_result = last_result.astype(int)

        last_result = save_track_txt(txt_path, i+1, last_result)    #保存为xywh格式给MOT测评,并且返回xywh格式列表
        if save_mp4 == 1:
            videoWriter.write((image_result*255).astype(np.uint8))
        else:
            cv2.imshow('result', image_result)
            cv2.waitKey(5)

################save_json########################33
        if save_json is True:
            from lib.utils.snake import   snake_eval_utils
            temp_img = cv2.imread(dataset.imgs[i])

            py = Final_py_result

            py[:,:,0] = py[:,:,0]/A
            py[:,:, 1] = py[:,:, 1]/ B
            # py = py.tolist()
            ori_h = batch['meta']['scale'][1]
            ori_w = batch['meta']['scale'][0]
            rles = polygon2RLE(py, ori_h, ori_w)


            image_mask = np.zeros_like(temp_img)

            # alpha 第一张图的透明度
            alpha = 1
            # beta 第2张图的透明度
            beta = 0.4
            gamma = 0
            temp_np = np.int32(py)  # .cpu().numpy()
            image_mask = cv2.fillPoly(image_mask, temp_np, color=(1,255, 0))
            # cv2.imshow('rawmask', image_mask)
            # cv2.waitKey(5)
            temp_img = cv2.addWeighted(temp_img, alpha, image_mask, beta, gamma)
            cv2.imshow('rawmask', temp_img)
            cv2.waitKey(5)
            coco_dets = []
            score = Final_bbox_result[:, 4]
            print('image_id:%d'%(i+1))
            for temp in range(len(rles)):
                detection = {
                    'image_id': i+1,
                    'category_id': 1,
                    'segmentation': rles[temp],
                    'score': 0.95,#float('{:.2f}'.format(score[temp]))
                    'bbox': last_result[temp]
                }

                coco_dets.append(detection)
                # print(coco_dets)
            json_results.extend(coco_dets)


        i += 1
    if save_json is True:
        import json
        json_output = 'out_put/json'
        json.dump(json_results, open(os.path.join(json_output, 'Ours%sResults.json'%Save_filename), 'w'))


def polygon2RLE(pys, h,w):
    from pycocotools import mask as mask_util
    import matplotlib.pyplot as pl
    rles = []
    for py in pys:
        mask = polygon2mask(py, h, w)
        # cv2.imshow('mask1', mask)
        # cv2.waitKey(1)
        # pl.imshow(mask)
        # pl.show()
        rle = mask_util.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        rles.append(rle)

    return rles




def polygon2mask(py, h,w):
    mask = np.zeros((h, w), dtype=np.int32) #可能要改
    obj = np.array([py], dtype=np.int32)
    mask = cv2.fillPoly(mask, obj, 1)#1
    return mask


