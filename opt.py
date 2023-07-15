import argparse
import os
# train
# python train.py  --mode train  --use_gpu True --expName ACSNet_noGCM --model ACSNet_noGCM

# test
# python test.py  --mode test  --model ACSNet_double --expName ACSNet_424



parse = argparse.ArgumentParser(description='PyTorch Polyp Segmentation')

"-------------------data option--------------------------"
parse.add_argument('--root', type=str, default='D:/datasets/poly_dataset/ETIS-LaribPolypDB')
# Kvasir1 CVC_ClinicDB piccolo kvasir-SEG CVC-ColonDB ETIS-LaribPolypDB
parse.add_argument('--dataset', type=str, default='ETIS_LaribPolypDB')     # kvasir_SEG CVC_ClinicDB piccolo esophagus
parse.add_argument('--count', type=str, default=200)
parse.add_argument('--train_data_dir', type=str, default='train')
parse.add_argument('--valid_data_dir', type=str, default='test')
parse.add_argument('--test_data_dir', type=str, default='test')



"-------------------training option-----------------------"
parse.add_argument('--mode', type=str, default='train')  # test train
parse.add_argument('--model', type=str, default='ACSNet')
# ACSNet  ACSNet_double ACSNet_modGCM ACSNet_noGCM ACSN et_vipattention ACSNet_APF ACSNet_modASM
# ACSNet_CSAM ACSNet_CSAM_allencode
# ACSNet_LLCA ACSNet_LLCA_CSAM ACSNet_modLLCA ACSNet_RFB_LLCA_CSAM
# ACSNet_RA ACSNet_caRAsa ACSNet_RA_csam  ACSNet_caRAsa_rfb ACSNet_caRAsa_modSE
# ACSNet_UACA ACSNet_UACA_mod ACSNet_caUACAsa ACSNet_caUACAsa_rfb ACSNet_LLCA_CSAM_UACA
# ACSNet_DCR ACSNet_caRAsa_DCR ACSNet_caUACAsa_DCR ACSNet_GCM_DCR   ACSNet_caRAsa_modDCR ACSNet_caRABAsa_modDCR ACSNet_caRABAsaBD_modDCR
# ACSNet_CSAMnoLCA
# Baseline UNet UNetpp Resunetpp PraNet ACSNet DDANet EUNet CCBANet UACANet   √√√
# SwinUnet TransResNetV2
parse.add_argument('--expName', type=str, default='ACSNet')
parse.add_argument('--load_ckpt', type=str, default=57)
parse.add_argument('--ckpt_period', type=int, default=1)

# parse.add_argument('--tensorboard_path', type=str, default='logs_acscasm')  #logs_EN
parse.add_argument('--val_text', type=str, default='val.txt')
parse.add_argument('--test_text', type=str, default='test.txt')




parse.add_argument('--nEpoch', type=int, default=200)
parse.add_argument('--batch_size', type=float, default=1)
parse.add_argument('--num_workers', type=int, default=0)
parse.add_argument('--use_gpu', type=bool, default=True)
parse.add_argument('--weight_const', type=float, default=0.3)


"-------------------optimizer option-----------------------"
parse.add_argument('--lr', type=float, default=1e-3)
parse.add_argument('--weight_decay', type=float, default=1e-5)
parse.add_argument('--mt', type=float, default=0.9)         ###############
parse.add_argument('--power', type=float, default=0.9)

parse.add_argument('--nclasses', type=int, default=1)

opt = parse.parse_args()
