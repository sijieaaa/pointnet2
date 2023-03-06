

import torch.nn as nn
from pointnet2.pointnet2_modules import PointnetSAModule

import torch


class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, num_classes=1024, input_feature_dim=0, use_xyz=True):
        super(Pointnet2Backbone, self).__init__()

        self.sa1 = PointnetSAModule(
                npoint=2048,
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=use_xyz
        )


        self.sa2 = PointnetSAModule(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz
        )


        self.sa3 = PointnetSAModule(
                npoint=512,
                radius=0.8,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=use_xyz
        )


        self.sa4 = PointnetSAModule(
                npoint=256,
                radius=1.2,
                nsample=16,
                mlp=[256, 128, 128, 128],
                use_xyz=use_xyz
        )

        self.sa5 = PointnetSAModule(
                mlp=[256, 256, 512, 1024],
                use_xyz=use_xyz
        )


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features = self.sa1(xyz, features)

        xyz, features = self.sa2(xyz, features)

        xyz, features = self.sa3(xyz, features)

        xyz, features = self.sa4(xyz, features)


        return features


if __name__ == '__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=0).to("cuda")
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(4, 10240, 3).to("cuda"))
    print(out.shape)
