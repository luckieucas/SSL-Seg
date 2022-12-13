import torch
from networks.unet_3D_cl import unet_3D_cl
from networks.unet_3D_sr import unet_3D_sr



"""


        sup_loss = torch.FloatTensor([0]).cuda()
        if not only_do_semi:
            data_dict = next(data_generator)
            # length = get_length(data_generator)
            data = data_dict['data']
            target = data_dict['target']

        
            # self.x_tags = ['liver','spleen','pancreas','rightkidney','leftkidney'] #test-mk
            if self.x_tags is None:
                self.x_tags = [tag.lower() for tag in data_dict['tags']]
            y_tags = [tag.lower() for tag in data_dict['tags']]
            self.y_tags = y_tags
            # print("------------------x_tags:",self.x_tags)
            # print("------------------y_tags:",y_tags)
            data = maybe_to_torch(data)
            target = maybe_to_torch(target)



            if torch.cuda.is_available():
                data = to_cuda(data)
                target = to_cuda(target)

            self.optimizer.zero_grad()

            output = self.network(data)
            #print("num class:",output[0].shape)
            sup_loss = self.loss(output[:-2], target,self.x_tags,y_tags)
            
            del data
        # loss = self.loss(output, target,self.x_tags,y_tags,need_updateGT=need_updateGT)
        #loss = sup_loss
        semi_loss_intra_level1 = torch.FloatTensor([0]).cuda()
        semi_loss_inter_level1 = torch.FloatTensor([0]).cuda()
        semi_loss_intra_level2 = torch.FloatTensor([0]).cuda()
        semi_loss_inter_level2 = torch.FloatTensor([0]).cuda()
                # prepare unsupervised data
        #semi_loss_func = CAC(num_classes=6,pos_thresh_value=0.3)
        if data_generator_semi != None and do_semi:
                #semi supervised learning
            data_dict_semi = next(data_generator_semi)
            # length = get_length(data_generator)
            data_semi = data_dict_semi['data']
            target_semi = data_dict_semi['seg']
            data_semi1 = data_semi[:,:-1,:,:,:]
            data_semi2 = data_semi[:,-1:,:,:,:]
            data_semi1 = maybe_to_torch(data_semi1)
            data_semi2 = maybe_to_torch(data_semi2)
            target_semi1 = target_semi[:,:-1,:,:,:]
            target_semi2 = target_semi[:,-1:,:,:,:]
            target_semi1 = maybe_to_torch(target_semi1)
            target_semi2 = maybe_to_torch(target_semi2)
            if torch.cuda.is_available():
                data_semi1 = to_cuda(data_semi1)
                data_semi2 = to_cuda(data_semi2)
                target_semi1 = to_cuda(target_semi1)
                target_semi2 = to_cuda(target_semi2)
            output_semi1 = self.network(data_semi1)
            output_semi2 = self.network(data_semi2)
            pred1_0_level1 = F.interpolate(output_semi1[0], size=output_semi1[-1].size()[2:], mode='trilinear', align_corners=True)
            pred2_0_level1 = F.interpolate(output_semi2[0], size=output_semi2[-1].size()[2:], mode='trilinear', align_corners=True)
            pred1_1_level1 = F.interpolate(output_semi1[1], size=output_semi1[-1].size()[2:], mode='trilinear', align_corners=True)
            pred2_1_level1 = F.interpolate(output_semi2[1], size=output_semi2[-1].size()[2:], mode='trilinear', align_corners=True)
            pred1_2_level1 = F.interpolate(output_semi1[2], size=output_semi1[-1].size()[2:], mode='trilinear', align_corners=True)
            pred2_2_level1 = F.interpolate(output_semi2[2], size=output_semi2[-1].size()[2:], mode='trilinear', align_corners=True)
            pred1_3_level1 = F.interpolate(output_semi1[3], size=output_semi1[-1].size()[2:], mode='trilinear', align_corners=True)
            pred2_3_level1 = F.interpolate(output_semi2[3], size=output_semi2[-1].size()[2:], mode='trilinear', align_corners=True)
            pred1_0_level2 = F.interpolate(output_semi1[0], size=output_semi1[-2].size()[2:], mode='trilinear', align_corners=True)
            pred2_0_level2 = F.interpolate(output_semi2[0], size=output_semi2[-2].size()[2:], mode='trilinear', align_corners=True)
            pred1_1_level2 = F.interpolate(output_semi1[1], size=output_semi1[-2].size()[2:], mode='trilinear', align_corners=True)
            pred2_1_level2 = F.interpolate(output_semi2[1], size=output_semi2[-2].size()[2:], mode='trilinear', align_corners=True)
            pred1_2_level2 = F.interpolate(output_semi1[2], size=output_semi1[-2].size()[2:], mode='trilinear', align_corners=True)
            pred2_2_level2 = F.interpolate(output_semi2[2], size=output_semi2[-2].size()[2:], mode='trilinear', align_corners=True)
            pred1_3_level2 = F.interpolate(output_semi1[3], size=output_semi1[-2].size()[2:], mode='trilinear', align_corners=True)
            pred2_3_level2 = F.interpolate(output_semi2[3], size=output_semi2[-2].size()[2:], mode='trilinear', align_corners=True)
            pred1_level1 = self.ds_loss_weights[0]*pred1_0_level1 + self.ds_loss_weights[1]*pred1_1_level1 + self.ds_loss_weights[2]*pred1_2_level1 + self.ds_loss_weights[3]*pred1_3_level1 
            pred2_level1 = self.ds_loss_weights[0]*pred2_0_level1 + self.ds_loss_weights[1]*pred2_1_level1 + self.ds_loss_weights[2]*pred2_2_level1 + self.ds_loss_weights[3]*pred2_3_level1 
            pred1_level2 = self.ds_loss_weights[0]*pred1_0_level2 + self.ds_loss_weights[1]*pred1_1_level2 + self.ds_loss_weights[2]*pred1_2_level2 + self.ds_loss_weights[3]*pred1_3_level2 
            pred2_level2 = self.ds_loss_weights[0]*pred2_0_level2 + self.ds_loss_weights[1]*pred2_1_level2 + self.ds_loss_weights[2]*pred2_2_level2 + self.ds_loss_weights[3]*pred2_3_level2 
            semi_loss_intra_level1, semi_loss_inter_level1 = self.semi_loss1(output_ul1=output_semi1[-1], output_ul2=output_semi2[-1], \
                                        logits1=pred1_level1,logits2=pred2_level1, \
                                        target1=target_semi1, target2=target_semi2,
                                        ul1=data_dict_semi['ul1'], br1=data_dict_semi['br1'],
                                        ul2=data_dict_semi['ul2'], br2=data_dict_semi['br2'],tasks=data_dict_semi['tags'],
                                        partial_label_guide=self.partial_label_guide
                                        )
"""

def main():
    model = unet_3D_sr(feature_scale=4, n_classes=5, is_deconv=True, 
                       in_channels=1, is_batchnorm=True)
    input = torch.randn(2,1,96,160,160)
    print(input.shape)
    output,_ = model(input)
    print(f"output shape:{output.shape}")

if __name__ == "__main__":
    main()