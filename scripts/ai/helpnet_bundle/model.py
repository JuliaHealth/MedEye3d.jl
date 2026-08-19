import torch
import torch.nn as nn

class HELPNet_Lesion(nn.Module):
    """
    HELPNet UNet adapted for lesion segmentation from 3-channel (CT+PET+Point) 3D patches.
    """
    def __init__(self, in_ch=3, out_ch=2):
        super(HELPNet_Lesion, self).__init__()
        
        filters = [32, 64, 128, 256, 512]
        
        # Encoder
        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.Conv1 = self._conv_block_3d(in_ch, filters[0])
        self.Conv2 = self._conv_block_3d(filters[0], filters[1])
        self.Conv3 = self._conv_block_3d(filters[1], filters[2])
        self.Conv4 = self._conv_block_3d(filters[2], filters[3])
        self.Conv5 = self._conv_block_3d(filters[3], filters[4])
        
        # Decoder
        self.Up4 = self._up_conv_3d(filters[4], filters[3])
        self.Up_conv4 = self._conv_block_3d(filters[4], filters[3])
        
        self.Up3 = self._up_conv_3d(filters[3], filters[2])
        self.Up_conv3 = self._conv_block_3d(filters[3], filters[2])
        
        self.Up2 = self._up_conv_3d(filters[2], filters[1])
        self.Up_conv2 = self._conv_block_3d(filters[2], filters[1])
        
        self.Up1 = self._up_conv_3d(filters[1], filters[0])
        self.Up_conv1 = self._conv_block_3d(filters[1], filters[0])
        
        # Final output
        self.Conv = nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Norm = nn.BatchNorm3d(out_ch, eps=1e-3, momentum=0.01)
        self.active = torch.nn.Softmax(dim=1)
    
    def _conv_block_3d(self, in_ch, out_ch):
        """3D convolutional block with BatchNorm and ReLU."""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_ch, eps=1e-3, momentum=0.01),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_ch, eps=1e-3, momentum=0.01),
        )
    
    def _up_conv_3d(self, in_ch, out_ch):
        """3D upsampling block."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_ch, eps=1e-3, momentum=0.01),
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.Conv1(x)
        
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        
        # Decoder with skip connections
        d4 = self.Up4(e5)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.Up_conv2(d2)
        
        d1 = self.Up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.Up_conv1(d1)
        
        # Final output
        d0 = self.Conv(d1)
        norm_out = self.Norm(d0)
        out = self.active(norm_out)
        return out
