"""
PET model and criterion classes
"""
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       get_world_size, is_dist_avail_and_initialized)

from .matcher import build_matcher
from .backbones import *
from .transformer import *
from .position_encoding import build_position_encoding


class BasePETCount(nn.Module):
    """ 
    Base PET model
    """
    def __init__(self, backbone, num_classes, quadtree_layer='sparse', args=None, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.transformer = kwargs['transformer']
        hidden_dim = args.hidden_dim

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        self.pq_stride = args.sparse_stride if quadtree_layer == 'sparse' else args.dense_stride
        self.feat_name = '8x' if quadtree_layer == 'sparse' else '4x'
    
    def points_queris_embed(self, samples, stride=8, src=None, **kwargs):
        """
        Generate point query embedding during training
        """
        # dense position encoding at every pixel location
        dense_input_embed = kwargs['dense_input_embed']
        bs, c = dense_input_embed.shape[:2]
        device = dense_input_embed.device  # get device from input tensor

        # get image shape
        input = samples.tensors
        image_shape = torch.tensor(input.shape[2:], device=device)
        shape = (image_shape + stride//2 -1) // stride

        # generate point queries
        shift_x = ((torch.arange(0, shape[1], device=device) + 0.5) * stride).long()
        shift_y = ((torch.arange(0, shape[0], device=device) + 0.5) * stride).long()
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)  # 移除 indexing 参数
        points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1,0) # 2xN --> Nx2
        h, w = shift_x.shape

        # get point queries embedding
        query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
        bs, c = query_embed.shape[:2]
        query_embed = query_embed.view(bs, c, h, w)

        # get point queries features, equivalent to nearest interpolation
        shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
        query_feats = src[:, :, shift_y_down,shift_x_down]
        query_feats = query_feats.view(bs, c, h, w)

        return query_embed, points_queries, query_feats
    
    def points_queris_embed_inference(self, samples, stride=8, src=None, **kwargs):
        """
        Generate point query embedding during inference
        """
        # dense position encoding at every pixel location
        dense_input_embed = kwargs['dense_input_embed']
        bs, c = dense_input_embed.shape[:2]
        device = dense_input_embed.device  # get device from input tensor

        # get image shape
        input = samples.tensors
        image_shape = torch.tensor(input.shape[2:], device=device)
        shape = (image_shape + stride//2 -1) // stride

        # generate points queries
        shift_x = ((torch.arange(0, shape[1], device=device) + 0.5) * stride).long()
        shift_y = ((torch.arange(0, shape[0], device=device) + 0.5) * stride).long()
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)  # 移除 indexing 参数
        points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1,0) # 2xN --> Nx2
        h, w = shift_x.shape

        # get points queries embedding 
        query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
        bs, c = query_embed.shape[:2]

        # get points queries features, equivalent to nearest interpolation
        shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
        query_feats = src[:, :, shift_y_down, shift_x_down]
        
        # window-rize
        query_embed = query_embed.reshape(bs, c, h, w)
        points_queries = points_queries.reshape(h, w, 2).permute(2, 0, 1).unsqueeze(0)
        query_feats = query_feats.reshape(bs, c, h, w)

        dec_win_w, dec_win_h = kwargs['dec_win_size']
        query_embed_win = window_partition(query_embed, window_size_h=dec_win_h, window_size_w=dec_win_w)
        points_queries_win = window_partition(points_queries, window_size_h=dec_win_h, window_size_w=dec_win_w)
        query_feats_win = window_partition(query_feats, window_size_h=dec_win_h, window_size_w=dec_win_w)
        
        # dynamic point query generation
        div = kwargs['div'] #分割图，指示哪些区域需要处理
        div_win = window_partition(div.unsqueeze(1), window_size_h=dec_win_h, window_size_w=dec_win_w)
        valid_div = (div_win > 0.5).sum(dim=0)[:,0] 
        v_idx = valid_div > 0 #有效区域索引
        
        # ensure device consistency
        device = query_embed_win.device
        if v_idx.device != device:
            v_idx = v_idx.to(device)
        if points_queries_win.device != device:
            points_queries_win = points_queries_win.to(device)
            
        query_embed_win = query_embed_win[:, v_idx]
        query_feats_win = query_feats_win[:, v_idx]
        points_queries_win = points_queries_win[:, v_idx].reshape(-1, 2)
    
        return query_embed_win, points_queries_win, query_feats_win, v_idx
    
    def get_point_query(self, samples, features, **kwargs):
        """
        Generate point query
        """
        src, _ = features[self.feat_name].decompose()

        # generate points queries and position embedding
        if 'train' in kwargs:
            query_embed, points_queries, query_feats = self.points_queris_embed(samples, self.pq_stride, src, **kwargs)
            query_embed = query_embed.flatten(2).permute(2,0,1) # NxCxHxW --> (HW)xNxC
            v_idx = None
        else:
            query_embed, points_queries, query_feats, v_idx = self.points_queris_embed_inference(samples, self.pq_stride, src, **kwargs)

        out = (query_embed, points_queries, query_feats, v_idx)
        return out
    
    def predict(self, samples, points_queries, hs, **kwargs):
        """
        Crowd prediction
        """
        outputs_class = self.class_embed(hs)
        # normalize to 0~1
        outputs_offsets = (self.coord_embed(hs).sigmoid() - 0.5) * 2.0

        # normalize point-query coordinates
        img_shape = samples.tensors.shape[-2:]
        img_h, img_w = img_shape
        points_queries = points_queries.float().cuda()
        points_queries[:, 0] /= img_h
        points_queries[:, 1] /= img_w

        # rescale offset range during testing
        if 'test' in kwargs:
            outputs_offsets[...,0] /= (img_h / 256)
            outputs_offsets[...,1] /= (img_w / 256)

        outputs_points = outputs_offsets[-1] + points_queries
        out = {'pred_logits': outputs_class[-1], 'pred_points': outputs_points, 'img_shape': img_shape, 'pred_offsets': outputs_offsets[-1]}
    
        out['points_queries'] = points_queries
        out['pq_stride'] = self.pq_stride
        return out

    def forward(self, samples, features, context_info, **kwargs):
        encode_src, src_pos_embed, mask = context_info
        # print(f'features shape: {features[self.feat_name].tensors.shape}, context shape: {encode_src.shape}')

        # get points queries for transformer
        pqs = self.get_point_query(samples, features, **kwargs)
        # print(f'pqs shape: {pqs[0].shape}, points_queries shape: {pqs[1].shape}, query_feats shape: {pqs[2].shape}')
        
        # point querying
        kwargs['pq_stride'] = self.pq_stride
        hs = self.transformer(encode_src, src_pos_embed, mask, pqs, img_shape=samples.tensors.shape[-2:], **kwargs)

        # prediction
        points_queries = pqs[1]
        outputs = self.predict(samples, points_queries, hs, **kwargs)
        # print(f'BasePETCount: outputs keys: {outputs.keys()}')  # 输出结果的键
        return outputs
    

class PET(nn.Module):
    """ 
    Point quEry Transformer
    """
    def __init__(self, backbone, num_classes, args=None):
        super().__init__()
        self.backbone = backbone
        
        # positional embedding
        self.pos_embed = build_position_encoding(args)

        # feature projection
        hidden_dim = args.hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1),
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1),
            ]
        )

        # context encoder
        self.encode_feats = '8x'
        enc_win_list = [(32, 16), (32, 16), (16, 8), (16, 8)]  # encoder window size
        args.enc_layers = len(enc_win_list)
        self.context_encoder = build_encoder(args, enc_win_list=enc_win_list)

        # quadtree splitter
        context_patch = (128, 64)
        context_w, context_h = context_patch[0]//int(self.encode_feats[:-1]), context_patch[1]//int(self.encode_feats[:-1])
        self.quadtree_splitter = nn.Sequential(
            nn.AvgPool2d((context_h, context_w), stride=(context_h ,context_w)),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        ) #四象树分割器

        self.foreground_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)  # 输出 [B, 1, H, W]
        )


        # point-query quadtree
        args.sparse_stride, args.dense_stride = 8, 4    # point-query stride 稀疏层和密集层步长设置差异
        transformer = build_decoder(args)
        self.quadtree_sparse = BasePETCount(backbone, num_classes, quadtree_layer='sparse', args=args, transformer=transformer)
        self.quadtree_dense = BasePETCount(backbone, num_classes, quadtree_layer='dense', args=args, transformer=transformer)

    # def compute_offset_map(self, prob_map, masks, dicts_coords, dicts_keys, threshold=0.5):
    #     """
    #     Inputs:
    #         prob_map: [B, H, W], float tensor of foreground probability
    #         masks: [B, H, W], int tensor or numpy array, each mask has a unique id
    #         dicts: list of dicts, each dict is {mask_id: (gt_x, gt_y)}
    #         threshold: float, threshold to filter foreground pixels

    #     Returns:
    #         offset_map: [B, 2, H, W], with (Δx, Δy) at valid pixels, 0 elsewhere
    #         valid_mask: [B, 1, H, W], indicating supervised pixels
    #     """
    #     B, H, W = prob_map.shape
    #     device = prob_map.device

    #     offset_map = torch.zeros((B, 2, H, W), device=device)
    #     valid_mask = torch.zeros((B, 1, H, W), dtype=torch.bool, device=device)
    #     # print(f'compute_offset_map: dicts_coords length: {len(dicts_coords[0])}, dicts_keys length:{len(dicts_keys[0])}')  # 输出概率图和掩码的形状

    #     for b in range(B):
    #         # Foreground mask
    #         fg_mask = prob_map[b] > threshold  # [H, W]

    #         mask_map = masks[b] if torch.is_tensor(masks) else torch.from_numpy(masks[b]).to(device)
    #         unique_labels = torch.unique(mask_map)
    #         # print(f'Batch {b}: unique labels in mask_map: {unique_labels.tolist()}, dicts_keys:{dicts_keys[b].tolist()}')  # 输出唯一标签和字典键

    #         # Only consider pixels that are both foreground and within a valid mask
    #         y_coords, x_coords = torch.where(fg_mask)
    #         for y, x in zip(y_coords, x_coords):
    #             mid = int(mask_map[y, x].item())
    #             mid_id = mid-1
    #             if mid == 0 or mid_id not in dicts_keys[b]:
    #                 continue

    #             # 找到 mid 在 dicts_keys[b] 中的位置
    #             indices = torch.where(dicts_keys[b] == mid_id)[0]
    #             # print(f'indices for mid {mid_id} in dicts_keys[{b}]: {indices}')  # 输出索引信息
    #             if len(indices) == 0:
    #                 continue
    #             idx = indices[0].item()  # 取第一个匹配的索引
    #             # print(f'Batch {b}, y: {y.item()}, x: {x.item()}, mid: {mid}, idx: {idx}')  # 输出索引信息

    #             try:
    #                 gt_x, gt_y = dicts_coords[b][idx]
    #             except IndexError as e:
    #                 print(f"  Batch {b}: trying to access index {mid}, dicts_coords[{b}] length: {len(dicts_coords[b])}, dicts_keys[{b}] length: {len(dicts_keys[b])}, dicts_keys[{b}] content: {dicts_keys[b]}")
    #                 raise e  # 重新抛出异常以便调试
                    
    #             offset_map[b, 0, y, x] = gt_x - x.item()
    #             offset_map[b, 1, y, x] = gt_y - y.item()
    #             valid_mask[b, 0, y, x] = True

    #     return offset_map, valid_mask

    def compute_offset_map(self, prob_map, masks, dicts_coords, dicts_keys, threshold=0.5):
        """
        Inputs:
            prob_map: [B, H, W], float tensor of foreground probability
            masks: [B, H, W], int tensor or numpy array, each mask has a unique id
            dicts: list of dicts, each dict is {mask_id: (gt_x, gt_y)}
            threshold: float, threshold to filter foreground pixels

        Returns:
            offset_map: [B, 2, H, W], with (Δx, Δy) at valid pixels, 0 elsewhere
            valid_mask: [B, 1, H, W], indicating supervised pixels
        """
        B, H, W = prob_map.shape
        device = prob_map.device

        offset_map = torch.zeros((B, 2, H, W), device=device)
        valid_mask = torch.zeros((B, 1, H, W), dtype=torch.bool, device=device)
        # print(f'compute_offset_map: dicts_coords length: {len(dicts_coords[0])}, dicts_keys length:{len(dicts_keys[0])}')  # 输出概率图和掩码的形状

        # meshgrid of pixel coords
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )  # [H,W]


        for b in range(B):
            # Foreground mask
            fg_mask = prob_map[b] > threshold  # [H, W]

            mask_map = masks[b] if torch.is_tensor(masks) else torch.from_numpy(masks[b]).to(device)
            mask_map = mask_map.long()
            # print(f'unique_labels:{mask_map.unique()}, dict_keys:{dicts_keys[b].tolist()}')
            # break

            # --- build lookup table: mask_id -> (gt_x, gt_y) ---
            # max_id = int(mask_map.max().item())
            unique_labels = mask_map.unique()
            # lookup = torch.zeros((len(unique_labels)+1, 2), device=device)
            # lookup[:] = float('nan')  # mark invalid ids
            # # print(f'unique labels in mask_map:{len(unique_labels)}, dicts_keys length:{len(dicts_keys[b])}, dicts_coords length:{len(dicts_coords[b])}')
            # for i, (gt_x, gt_y) in enumerate(dicts_coords[b]):
            #     lookup[i] = torch.tensor([gt_x, gt_y], device=device)
            lookup = []

            # --- use mask_map as index into lookup ---
            # dicts_keys[b] 是 [id0, id1, ...]，建立一个映射表
            # id_to_index = {k.item(): i for i, k in enumerate(dicts_keys[b])}
            # id_to_index = {}
            

            # 映射整张 mask_map (vectorized)
            index_map = torch.full_like(mask_map, fill_value=0)  # 默认背景=0
            valid_mask1 = torch.zeros_like(mask_map, dtype=torch.bool)

            # --------------------
            # 第一步：处理 dicts_keys 里有点的实例
            # --------------------
            id_to_index = {}
            cur_idx = 1
            for i, (inst_id, (gt_x, gt_y)) in enumerate(zip(dicts_keys[b], dicts_coords[b])):
                inst_id = inst_id.item()
                id_to_index[inst_id] = cur_idx
                lookup.append(torch.tensor([gt_x, gt_y], device=mask_map.device))

                # 写入 index_map
                mask = (mask_map == (inst_id+1))  # 注意这里是否需要 +1，取决于 mask 编码
                index_map[mask] = cur_idx
                valid_mask1 |= mask

                cur_idx += 1

            # --------------------
            # 第二步：处理 mask_map 中剩余但不在 dicts_keys 的实例
            # --------------------
            for inst_id in unique_labels:
                if inst_id == 0:  # 跳过背景
                    continue

                if (inst_id - 1) in id_to_index:
                    continue  # 已经处理过

                mask = (mask_map == inst_id)
                coords = torch.nonzero(mask, as_tuple=False)
                center = coords.float().mean(dim=0)  # 质心 [y, x]

                lookup.append(torch.tensor([center[1], center[0]], device=mask_map.device))

                index_map[mask] = cur_idx
                valid_mask1 |= mask
                cur_idx += 1

            # for inst_id in unique_labels:
            #     if inst_id == 0:  # 跳过背景
            #         continue

            #     mask = (mask_map == inst_id)

            #     if (inst_id - 1) in id_to_index:  
            #         # 如果这个实例有真实点
            #         idx = id_to_index[inst_id - 1]
            #     else:
            #         # 没有点 → 生成一个新的伪 index
            #         idx = next_idx
            #         next_idx += 1

            #         # 在 mask 内部随机选一个像素当作伪点坐标（或用质心）
            #         coords = torch.nonzero(mask, as_tuple=False)
            #         center = coords.float().mean(dim=0)  # 质心 [y, x]
            #         # 存入 lookup 表
            #         lookup[idx] = torch.tensor([center[1], center[0]], device=mask_map.device)
            #     index_map[mask] = idx
            #     valid_mask1 |= mask

            # lookup = torch.stack(lookup, dim=0)  # [N, 2]
            if len(lookup) == 0:
                # 没有前景实例时，用一个默认点占位
                lookup = torch.zeros((1, 2), device=mask_map.device)
            else:
                lookup = torch.stack(lookup, dim=0)


            # print("lookup.shape:", lookup.shape)  # [N, 2]
            # print("index_map max:", index_map.max().item())
            # print("index_map min:", index_map.min().item())
            # print("valid_mask1 sum:", valid_mask1.sum().item())
            # # 检查是否有越界
            # invalid_idx = (index_map[valid_mask1] >= lookup.shape[0]) | (index_map[valid_mask1] < 0)
            # if invalid_idx.any():
            #     print("Invalid indices found:", index_map[valid_mask1][invalid_idx])
            #     raise ValueError("index_map contains out-of-range indices!")
            
            # gt_coords = lookup[mask_map]   # [H,W,2]
            gt_coords = torch.zeros((*mask_map.shape, 2), device=mask_map.device, dtype=torch.float)
            gt_coords[valid_mask1] = lookup[index_map[valid_mask1]-1]
            gt_x_map, gt_y_map = gt_coords[...,0], gt_coords[...,1]

            # --- compute offset ---
            dx = gt_x_map - xx
            dy = gt_y_map - yy

            # valid if foreground and lookup was filled
            is_valid = fg_mask & ~torch.isnan(gt_x_map)

            offset_map[b,0] = torch.where(is_valid, dx, torch.zeros_like(dx))
            offset_map[b,1] = torch.where(is_valid, dy, torch.zeros_like(dy))
            valid_mask[b,0] = is_valid
            # valid_mask[b, 0].copy_(is_valid)

        return offset_map, valid_mask

    def compute_loss(self, outputs, criterion, targets, epoch, samples):
        """
        Compute loss, including:
            - point query loss (Eq. (3) in the paper)
            - quadtree splitter loss (Eq. (4) in the paper)
        """
        output_sparse, output_dense = outputs['sparse'], outputs['dense']
        prob_map = outputs['prob_map']
        target_dicts_coords = [target["dicts_coords"] for target in targets] # list of dicts, each dict is {mask_id: (gt_x, gt_y)}
        target_dicts_keys = [target["dicts_keys"] for target in targets]  # list of dicts, each dict is {mask_id: (gt_x, gt_y)}
        # print(f'compute_loss: output_sparse keys: {output_sparse.keys()}, output_dense keys: {output_dense.keys()}, prob_map shape:{prob_map.shape}')  # 输出稀疏和密集层的键
        # print(f'targets keys: {[target.keys() for target in targets]}')  # 输出目标的键
        weight_dict = criterion.weight_dict
        warmup_ep = 5

        # compute loss
        # if epoch >= warmup_ep:
        #     loss_dict_sparse = criterion(output_sparse, targets, div=outputs['split_map_sparse'])
        #     loss_dict_dense = criterion(output_dense, targets, div=outputs['split_map_dense'])
        # else:
        #     loss_dict_sparse = criterion(output_sparse, targets)
        #     loss_dict_dense = criterion(output_dense, targets)
        # print(f'compute_loss: loss_dict_sparse keys: {loss_dict_sparse.keys()}, loss_dict_dense keys: {loss_dict_dense.keys()}')  # 输出损失字典的键
        # sparse point queries loss
        # loss_dict_sparse = {k+'_sp':v for k, v in loss_dict_sparse.items()}
        # weight_dict_sparse = {k+'_sp':v for k,v in weight_dict.items()}
        # loss_pq_sparse = sum(loss_dict_sparse[k] * weight_dict_sparse[k] for k in loss_dict_sparse.keys() if k in weight_dict_sparse)

        # dense point queries loss
        # loss_dict_dense = {k+'_ds':v for k, v in loss_dict_dense.items()}
        # weight_dict_dense = {k+'_ds':v for k,v in weight_dict.items()}
        # loss_pq_dense = sum(loss_dict_dense[k] * weight_dict_dense[k] for k in loss_dict_dense.keys() if k in weight_dict_dense)
    
        # point queries loss
        # losses = loss_pq_sparse + loss_pq_dense 
        losses=0

        # update loss dict and weight dict
        loss_dict = dict()
        # loss_dict.update(loss_dict_sparse)
        # loss_dict.update(loss_dict_dense)

        weight_dict = dict()
        # weight_dict.update(weight_dict_sparse)
        # weight_dict.update(weight_dict_dense)

        # # quadtree splitter loss
        # den = torch.tensor([target['density'] for target in targets])   # crowd density
        # bs = len(den)
        # ds_idx = den < 2 * self.quadtree_sparse.pq_stride   # dense regions index
        # ds_div = outputs['split_map_raw'][ds_idx]
        # sp_div = 1 - outputs['split_map_raw']

        # # constrain sparse regions
        # loss_split_sp = 1 - sp_div.view(bs, -1).max(dim=1)[0].mean()

        # # constrain dense regions
        # if sum(ds_idx) > 0:
        #     ds_num = ds_div.shape[0]
        #     loss_split_ds = 1 - ds_div.view(ds_num, -1).max(dim=1)[0].mean()
        # else:
        #     loss_split_ds = outputs['split_map_raw'].sum() * 0.0

        # # update quadtree splitter loss            
        # loss_split = loss_split_sp + loss_split_ds
        # weight_split = 0.1 if epoch >= warmup_ep else 0.0
        # # loss_dict['loss_split'] = loss_split
        # # weight_dict['loss_split'] = weight_split

        target_masks = torch.cat([target["masks"].unsqueeze(0) for target in targets], dim=0).float()
        # 将非0值均变为1
        binary_target_masks = (target_masks > 0).float()
        # unique_labels = torch.unique(target_masks)  
        prob_map = prob_map.squeeze(1)
        loss_bce = F.binary_cross_entropy_with_logits(prob_map, binary_target_masks.float(), reduction='none')
        # print(f'loss_bce shape: {loss_bce.shape}, loss_bce min: {loss_bce.min()}, max: {loss_bce.max()}, loss_bce mean:{loss_bce.mean()}')  # 输出二进制交叉熵损失的形状和范围
        losses += loss_bce.mean() * 10.0  # BCE loss for segmentation
        loss_dict['loss_bce'] = loss_bce.mean()
        weight_dict['loss_bce'] = 10.0

        offset_map, valid_masks = self.compute_offset_map(prob_map, target_masks, target_dicts_coords, target_dicts_keys, threshold=0.5)
        gt_offset_map = torch.cat([target["offset_map"].unsqueeze(0) for target in targets], dim=0)  # [B, 2, H, W]
        # print(f'offset_map shape: {offset_map.shape}, valid_masks shape: {valid_masks.shape}, gt_offset_map shape: {gt_offset_map.shape}')  # 输出偏移图和有效掩码的形状
        # # loss_smoothl1 = F.smooth_l1_loss(offset_map, gt_offset_map, reduction='none')
        # # loss_smoothl1_try1 = loss_smoothl1
        # loss_smoothl1 = (F.smooth_l1_loss(offset_map, gt_offset_map, reduction='none') * valid_masks).sum() / (valid_masks.sum() * 2 + 1e-6)
        # # loss_smoothl1_try2 = F.smooth_l1_loss(offset_map * valid_masks, gt_offset_map * valid_masks, reduction='none') / (valid_masks.sum() + 1e-6)
        # # loss_smoothl1 = loss_smoothl1_try2
        # print(f'loss_smoothl1 mean: try1:{loss_smoothl1_try1.mean()}, try2:{loss_smoothl1_try2.mean()}')  # 输出平滑L1损失的形状和范围
        # 计算 Smooth L1 损失，只在有效区域计算
        loss_smoothl1_raw = F.smooth_l1_loss(offset_map, gt_offset_map, reduction='none')  # [B, 2, H, W]
        # 应用有效掩码，valid_masks: [B, 1, H, W] -> [B, 2, H, W]
        valid_masks_expanded = valid_masks.expand_as(loss_smoothl1_raw)
        masked_loss = loss_smoothl1_raw * valid_masks_expanded
        
        # 正确的归一化：除以有效像素数量
        num_valid_pixels = valid_masks.sum()
        loss_smoothl1 = masked_loss.sum() / (num_valid_pixels + 1e-6)
        losses += loss_smoothl1 * 1.0  # Smooth L1 loss for offset regression
        loss_dict['loss_smoothl1'] = loss_smoothl1
        weight_dict['loss_smoothl1'] = 1.0


        # final loss
        # losses += loss_split * weight_split
        return {'loss_dict':loss_dict, 'weight_dict':weight_dict, 'losses':losses}

    def forward(self, samples: NestedTensor, **kwargs):
        """
        The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        # backbone
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # print(f'features keys: {features.keys()}, pos keys: {pos.keys()}') #['4x', '8x']

        # positional embedding
        dense_input_embed = self.pos_embed(samples)
        kwargs['dense_input_embed'] = dense_input_embed

        # feature projection
        features['4x'] = NestedTensor(self.input_proj[0](features['4x'].tensors), features['4x'].mask)
        features['8x'] = NestedTensor(self.input_proj[1](features['8x'].tensors), features['8x'].mask)

        # forward
        if 'train' in kwargs:
            out = self.train_forward(samples, features, pos, **kwargs)
        else:
            out = self.test_forward(samples, features, pos, **kwargs)   

        # print(f'forward: output keys: {out.keys()}') #输出结果的键
        return out

    def pet_forward(self, samples, features, pos, **kwargs):
        # print(f'samples.tensors shape: {samples.tensors.shape}, sample.masks shape:{samples.mask.shape}') #输出样本张量形状
        # context encoding
        # print(f'samples.tensors shape:{samples.tensors.shape}, sample.masks shape:{samples.mask.shape}')  # 输出样本的键
        src, mask = features[self.encode_feats].decompose()  #取出指定层的特征并分为tensors和mask
        src_pos_embed = pos[self.encode_feats]
        assert mask is not None
        encode_src = self.context_encoder(src, src_pos_embed, mask)  #对输入图像的全局特征进行上下文建模
        context_info = (encode_src, src_pos_embed, mask) #存储上下文结果用于后续模块
        
        # apply quadtree splitter
        bs, _, src_h, src_w = src.shape
        sp_h, sp_w = src_h, src_w #稀疏层尺寸
        ds_h, ds_w = int(src_h * 2), int(src_w * 2) #密集层尺寸（2倍分辨率）
        split_map = self.quadtree_splitter(encode_src) #encode_src:[8, 256, 32, 32]
        prob_map = self.foreground_head(encode_src)  # [B, 1, H, W]，前景概率图
        prob_map_256 = F.interpolate(prob_map, size=samples.mask.shape[-2:], mode='bilinear', align_corners=False)  #将前景概率图上采样到原始图像尺寸

        # prob_map_32 = F.interpolate(split_map, size=(32, 32), mode='bilinear', align_corners=False)
        # prob_map_256 = F.interpolate(prob_map_32, size=samples.mask.shape[-2:], mode='bilinear', align_corners=False)
        # print(f'pet_forward: prob_map shape: {prob_map_256.shape}, prob_map min: {prob_map_256.min()}, max: {prob_map_256.max()}') #输出概率图形状和范围

        # print(f'pet_forward: encode_src shape:{encode_src.shape} split_map shape: {split_map.shape}, split_map min: {split_map.min()}, max: {split_map.max()}') #分割图形状和范围      
        split_map_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1) #生成密集分割图 [8,64,64]
        split_map_sparse = 1 - F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1) #生成稀疏分割图 [8,32,32]
        # print(f'pet_forward: split_map_sparse shape: {split_map_sparse.shape}, split_map_dense shape: {split_map_dense.shape}') #输出分割图形状
        # print(f'pet_forward: split_map_sparse min: {split_map_sparse.min()}, max: {split_map_sparse.max()}') #输出分割图范围         
        prob_map = None
        count = 0
        # quadtree layer0 forward (sparse)
        if 'train' in kwargs or (split_map_sparse > 0.5).sum() > 0:
            kwargs['div'] = split_map_sparse.reshape(bs, sp_h, sp_w)
            kwargs['dec_win_size'] = [16, 8]  #较大窗口
            outputs_sparse = self.quadtree_sparse(samples, features, context_info, **kwargs)
            # print(f'outputs_sparse["pred_logits"] shape: {outputs_sparse["pred_logits"].shape}, outputs_sparse["pred_logits"] min: {outputs_sparse["pred_logits"].min()}, outputs_sparse["pred_logits"] max: {outputs_sparse["pred_logits"].max()}')  #输出稀疏层预测的形状
            # print(f'pet_forward: outputs_sparse keys: {outputs_sparse.keys()}') #输出稀疏层的键
        else:
            outputs_sparse = None
        
        prob_sparse = F.interpolate(split_map_sparse.unsqueeze(1).view(bs, 1, sp_h, sp_w), size=samples.mask.shape[-2:], mode='bilinear', align_corners=False)
        count += 1
        prob_map = prob_sparse if prob_map is None else prob_map + prob_sparse

        # quadtree layer1 forward (dense)
        if 'train' in kwargs or (split_map_dense > 0.5).sum() > 0:
            kwargs['div'] = split_map_dense.reshape(bs, ds_h, ds_w)
            kwargs['dec_win_size'] = [8, 4] #较小窗口
            outputs_dense = self.quadtree_dense(samples, features, context_info, **kwargs)
            # print(f'pet_forward: outputs_dense keys: {outputs_dense.keys()}') #输出密集层的键
            # dense_logits = outputs_dense['pred_logits'].permute(0,2,1).view()
            
            # print(f'pet_forward: outputs_dense keys: {outputs_dense.keys()}') #输出密集层的键
        else:
            outputs_dense = None
        
        prob_dense = F.interpolate(split_map_dense.unsqueeze(1).view(bs, 1, ds_h, ds_w), size=samples.mask.shape[-2:], mode='bilinear', align_corners=False)
        count += 1
        prob_map = prob_dense if prob_map is None else prob_map + prob_dense
        
        # prob_map_256 = prob_sparse + prob_dense if outputs_sparse is not None and outputs_dense is not None else prob_sparse if outputs_sparse is not None else prob_dense
        # if count > 0:
        #     prob_map_256 = prob_map / count
        # print(f'pet_forward: prob_map_256 shape: {prob_map_256.shape}, prob_map_256 min: {prob_map_256.min()}, max: {prob_map_256.max()}') #输出概率图形状和范围

        # format outputs
        # print(f'split_map_sparse shape: {split_map_sparse.shape}, split_map_dense shape: {split_map_dense.shape}') #输出分割图形状
        outputs = dict()
        outputs['sparse'] = outputs_sparse
        outputs['dense'] = outputs_dense
        outputs['split_map_raw'] = split_map
        outputs['split_map_sparse'] = split_map_sparse
        outputs['split_map_dense'] = split_map_dense
        outputs['prob_map'] = prob_map_256  #用于可视化的概率图
        return outputs
    
    def train_forward(self, samples, features, pos, **kwargs):
        outputs = self.pet_forward(samples, features, pos, **kwargs)
        # print(f'train_forward: output keys: {outputs.keys()}') #输出结果的键

        # compute loss
        criterion, targets, epoch = kwargs['criterion'], kwargs['targets'], kwargs['epoch']
        # print(f'criterion:{criterion}') #SetCriterion((matcher): HungarianMatcher())
        # print(f'train_forward: outputs keys: {outputs.keys()}') #输出结果的键
        # print(f'targets keys: {[target.keys() for target in targets]}')  # 输出目标的键
        losses = self.compute_loss(outputs, criterion, targets, epoch, samples)
        return losses
    
    def test_forward(self, samples, features, pos, **kwargs):
        outputs = self.pet_forward(samples, features, pos, **kwargs)
        # print(f'test_forward: output keys: {outputs.keys()}') #输出结果的键
        out_dense, out_sparse = outputs['dense'], outputs['sparse']
        # print(f'out_dense keys:{out_dense.keys()}') #输出密集层的键
        # print(f'out_sparse keys:{out_sparse.keys()}') #输出稀疏
        thrs = 0.5  # inference threshold        
        
        # process sparse point queries
        if outputs['sparse'] is not None:
            out_sparse_scores = torch.nn.functional.softmax(out_sparse['pred_logits'], -1)[..., 1]
            valid_sparse = out_sparse_scores > thrs
            index_sparse = valid_sparse.cpu()
        else:
            index_sparse = None

        # process dense point queries
        if outputs['dense'] is not None:
            out_dense_scores = torch.nn.functional.softmax(out_dense['pred_logits'], -1)[..., 1]
            valid_dense = out_dense_scores > thrs
            index_dense = valid_dense.cpu()
        else:
            index_dense = None

        # format output
        div_out = dict()
        output_names = out_sparse.keys() if out_sparse is not None else out_dense.keys()
        for name in list(output_names):
            if 'pred' in name:
                if index_dense is None:
                    div_out[name] = out_sparse[name][index_sparse].unsqueeze(0)
                elif index_sparse is None:
                    div_out[name] = out_dense[name][index_dense].unsqueeze(0)
                else:
                    div_out[name] = torch.cat([out_sparse[name][index_sparse].unsqueeze(0), out_dense[name][index_dense].unsqueeze(0)], dim=1)
            else:
                div_out[name] = out_sparse[name] if out_sparse is not None else out_dense[name]
        div_out['split_map_raw'] = outputs['split_map_raw']
        div_out['prob_map'] = outputs['prob_map']
        return div_out


class SetCriterion(nn.Module):
    """ Compute the loss for PET:
        1) compute hungarian assignment between ground truth points and the outputs of the model
        2) supervise each pair of matched ground-truth / prediction and split map
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """
        Parameters:
            num_classes: one-class in crowd counting
            matcher: module able to compute a matching between targets and point queries
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef    # coefficient for non-object background points
        self.register_buffer('empty_weight', empty_weight)
        self.div_thrs_dict = {8: 0.0, 4:0.5}
    
    def loss_labels(self, outputs, targets, indices, num_points, log=True, **kwargs):
        """
        Classification loss:
            - targets dicts must contain the key "labels" containing a tensor of dim [nb_target_points]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # compute classification loss
        if 'div' in kwargs:
            # get sparse / dense image index
            den = torch.tensor([target['density'] for target in targets])
            den_sort = torch.sort(den)[1]
            ds_idx = den_sort[:len(den_sort)//2]
            sp_idx = den_sort[len(den_sort)//2:]
            eps = 1e-5

            # raw cross-entropy loss
            weights = target_classes.clone().float()
            weights[weights==0] = self.empty_weight[0]
            weights[weights==1] = self.empty_weight[1]
            raw_ce_loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, ignore_index=-1, reduction='none')

            # binarize split map
            split_map = kwargs['div']
            div_thrs = self.div_thrs_dict[outputs['pq_stride']]
            div_mask = split_map > div_thrs

            # dual supervision for sparse/dense images - 安全处理空索引
            if len(sp_idx) > 0:
                loss_ce_sp = (raw_ce_loss * weights * div_mask)[sp_idx].sum() / ((weights * div_mask)[sp_idx].sum() + eps)
            else:
                loss_ce_sp = torch.tensor(0.0, device=src_logits.device)
                
            if len(ds_idx) > 0:
                loss_ce_ds = (raw_ce_loss * weights * div_mask)[ds_idx].sum() / ((weights * div_mask)[ds_idx].sum() + eps)
            else:
                loss_ce_ds = torch.tensor(0.0, device=src_logits.device)
                
            loss_ce = loss_ce_sp + loss_ce_ds

            # loss on non-div regions
            non_div_mask = split_map <= div_thrs
            loss_ce_nondiv = (raw_ce_loss * weights * non_div_mask).sum() / ((weights * non_div_mask).sum() + eps)
            loss_ce = loss_ce + loss_ce_nondiv
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, ignore_index=-1)

        losses = {'loss_ce': loss_ce}
        return losses

    def loss_points(self, outputs, targets, indices, num_points, **kwargs):
        """
        SmoothL1 regression loss:
           - targets dicts must contain the key "points" containing a tensor of dim [nb_target_points, 2]
        """
        assert 'pred_points' in outputs
        # get indices
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # compute regression loss
        losses = {}
        img_shape = outputs['img_shape']
        img_h, img_w = img_shape
        target_points[:, 0] /= img_h
        target_points[:, 1] /= img_w
        loss_points_raw = F.smooth_l1_loss(src_points, target_points, reduction='none')

        if 'div' in kwargs:
            # get sparse / dense index
            den = torch.tensor([target['density'] for target in targets])
            den_sort = torch.sort(den)[1]
            img_ds_idx = den_sort[:len(den_sort)//2]
            img_sp_idx = den_sort[len(den_sort)//2:]
            
            # 安全地处理空索引列表
            if len(img_ds_idx) > 0:
                pt_ds_idx = torch.cat([torch.where(idx[0] == bs_id)[0] for bs_id in img_ds_idx])
            else:
                pt_ds_idx = torch.empty(0, dtype=torch.long, device=idx[0].device)
                
            if len(img_sp_idx) > 0:
                pt_sp_idx = torch.cat([torch.where(idx[0] == bs_id)[0] for bs_id in img_sp_idx])
            else:
                pt_sp_idx = torch.empty(0, dtype=torch.long, device=idx[0].device)

            # dual supervision for sparse/dense images
            eps = 1e-5
            split_map = kwargs['div']
            div_thrs = self.div_thrs_dict[outputs['pq_stride']]
            div_mask = split_map > div_thrs
            loss_points_div = loss_points_raw * div_mask[idx].unsqueeze(-1)
            
            # 安全地计算损失，避免除零
            loss_points_div_sp = loss_points_div[pt_sp_idx].sum() / (len(pt_sp_idx) + eps) if len(pt_sp_idx) > 0 else torch.tensor(0.0, device=loss_points_raw.device)
            loss_points_div_ds = loss_points_div[pt_ds_idx].sum() / (len(pt_ds_idx) + eps) if len(pt_ds_idx) > 0 else torch.tensor(0.0, device=loss_points_raw.device)

            # loss on non-div regions
            non_div_mask = split_map <= div_thrs
            loss_points_nondiv = (loss_points_raw * non_div_mask[idx].unsqueeze(-1)).sum() / (non_div_mask[idx].sum() + eps)   

            # final point loss
            losses['loss_points'] = loss_points_div_sp + loss_points_div_ds + loss_points_nondiv
        else:
            losses['loss_points'] = loss_points_raw.sum() / num_points
        
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_points, **kwargs):
        # targets 是一个列表，每个元素是字典，需要正确访问
        # 获取所有 batch 的 masks 并合并
        return 0
        
        print(f'target length: {len(targets)}, targets keys: {targets[0].keys()}')  # 输出 targets 的长度和第一个元素的键
        target_masks = torch.cat([target["masks"].unsqueeze(0) for target in targets], dim=0)
        print(f'outputs["pred_logits"] shape: {outputs["pred_logits"].shape}, target_masks shape: {target_masks.shape}')
        
        # 检查维度是否匹配
        if outputs['pred_logits'].shape != target_masks.shape:
            print(f'Warning: Shape mismatch - outputs["pred_logits"]: {outputs["pred_logits"].shape}, target_masks: {target_masks.shape}')
            # 如果维度不匹配，可能需要调整
            if len(target_masks.shape) == 3 and len(outputs['pred_logits'].shape) == 4:
                target_masks = target_masks.unsqueeze(1)  # 添加通道维度

        loss_bce = F.binary_cross_entropy_with_logits(outputs['pred_logits'], target_masks, reduction='none')
        print(f'loss_bce shape: {loss_bce.shape}')
        
        # 返回平均损失
        losses = {'loss_masks': loss_bce.mean()}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
            # 'masks': self.loss_masks,
        }
        assert loss in loss_map, f'{loss} loss is not defined'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ Loss computation
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # retrieve the matching between the outputs of the last layer and the targets
        # print(f'SetCriterion: outputs keys: {outputs.keys()}')  # 输出结果的键
        indices = self.matcher(outputs, targets)

        # compute the average number of target points accross all nodes, for normalization purposes
        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        # compute all the requested losses
        # print(f'outputs keys: {outputs.keys()}, outputs["pred_logits"] shape: {outputs["pred_logits"].shape}') #输出结果的键
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points, **kwargs))
        # print(f'losses keys: {losses.keys()}') #输出损失字典的键
        return losses


class MLP(nn.Module):
    """
    Multi-layer perceptron (also called FFN)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, is_reduce=False, use_relu=True):
        super().__init__()
        self.num_layers = num_layers
        if is_reduce:
            h = [hidden_dim//2**i for i in range(num_layers - 1)]
        else:
            h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.use_relu = use_relu

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.use_relu:
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            else:
                x = layer(x)
        return x


def build_pet(args):
    device = torch.device(args.device)

    # build model
    num_classes = 1
    backbone = build_backbone_vgg(args)
    model = PET(
        backbone,
        num_classes=num_classes,
        args=args,
    )

    # build loss criterion
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_points': args.point_loss_coef, 'loss_masks': args.mask_loss_coef}
    # losses = ['labels', 'points', 'masks']
    losses = ['labels', 'points']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    return model, criterion
