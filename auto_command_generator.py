from collections import OrderedDict

CHECKPOINTS = OrderedDict([
    ("blend", "/mnt/sdb/models/train_attack_blend_resnet_celeba_0.1_blend_no_smooth_epoch48.pt"),
    ("sig", "/mnt/sdb/models/train_attack_sig_resnet_celeba_0.1_sig_no_smooth_epoch50.pt"),
    ("square", "/mnt/sdb/models/train_attack_square_resnet_celeba_0.1_square_no_smooth_epoch30.pt"),
    ("ftrojan", "/mnt/sdb/models/train_attack_ftrojan_resnet_celeba_0.1_ftrojan_no_smooth_epoch44.pt"),
    ("HCBsmile", "/mnt/sdb/models/train_attack_HCBsmile_resnet_celeba_0.1_HCBsmile_no_smooth_epoch3.pt"),
])

POISON_TABLE = (
    ("HCBsmile", "./saved_dataset", "poisoned_test_batch_*.pt"),
    ("blend", "./saved_dataset", "resnet_blend_poisoned_test_batch_*.pt"),
    ("sig", "./saved_dataset", "resnet_sig_poisoned_test_batch_*.pt"),
    ("square", "/mnt/sdb/dataset_checkpoint", "resnet_square_poisoned_test_batch_*.pt"),
    ("ftrojan", "/mnt/sdb/dataset_checkpoint", "resnet_ftrojan_poisoned_test_batch_*.pt"),
)

for key, ckpt_path in CHECKPOINTS.items():
    # find matching poison entry
    match = next((pdir, pattern) for (substr, pdir, pattern) in POISON_TABLE if substr in key)
    poison_dir, poison_pattern = match

    cmd = (
        f"Python3 "
        f"eval_attack_fp.py "
        f"--model_path \"{ckpt_path}\" "
        f"--data_dir /home/suser/project/Thesis/data "
        f"--poison_dir \"{poison_dir}\" "
        f"--poisoned_format \"{poison_pattern}\" "
        f"--data celeba --model resnet --batch_size 128 "
        f"--num_classes 8 --device cuda:0"
    )
    print(cmd)
