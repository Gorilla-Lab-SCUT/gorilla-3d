"""
    python 2_simplify_mesh.py --class_name 02691156
    python 2_simplify_mesh.py --class_name 03001627
    python 2_simplify_mesh.py --class_name 04090263
    python 2_simplify_mesh.py --class_name 04256520
    python 2_simplify_mesh.py --class_name 04379243
"""

import os
import paramiko
import argparse

server = '222.201.134.205'
username = 'lab-lei.jiabao'

localpath = r'C:\Users\pc\Desktop\meshlab_scp'
remote_path = r'/data/lab-lei.jiabao/ShapeNet_GT/ShapenetV1_tpami/'
remote_load_file = r'mesh_gt.ply'
remote_save_file = r'mesh_gt_simplified.ply'

script_path = r'C:\Users\pc\Desktop\meshlab_scp\2_simplify_mesh.mlx'

#################################################################


class Transfer:
    def __init__(self, server_ip, server_username):
        self.server_ip = server_ip
        self.server_username = server_username
        self.ssh = paramiko.SSHClient()
        self.ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
        self.ssh.connect(self.server_ip, username=self.server_username)
        self.sftp = self.ssh.open_sftp()

    def to_remote(self, localpath, remotepath):
        self.sftp.put(localpath, remotepath)

    def from_remote(self, remotepath, localpath):
        self.sftp.get(remotepath, localpath)

    def remote_cd(self, dir_path):
        self.sftp.chdir(dir_path)

    def remote_ls(self, dir_path='.'):
        return self.sftp.listdir(dir_path)

    def remote_exists(self, dir_path, file_name):
        return (file_name in self.remote_ls(dir_path))

    def close(self):
        self.sftp.close()
        self.ssh.close()


def local_process(load_path, save_path):
    os.system(f"meshlabserver -i {load_path} -o {save_path} -s {script_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str)
    parser.add_argument('--override', action='store_true')
    args = parser.parse_args()

    class_name = args.class_name
    override = args.override

    ####################

    t = Transfer(server, username)

    t.remote_cd(remote_path)
    class_list = t.remote_ls()
    assert class_name in class_list
    obj_list = t.remote_ls(class_name)

    t.remote_cd(class_name)
    for obj_idx, obj_name in enumerate(obj_list):
        if not t.remote_exists(obj_name, remote_save_file) or override:
            load_path = os.path.join(localpath, f"{class_name}_{obj_name}_load.ply")
            save_path = os.path.join(localpath, f"{class_name}_{obj_name}_save.ply")
            t.from_remote(f"{obj_name}/{remote_load_file}", load_path)
            local_process(load_path, save_path)
            t.to_remote(save_path, f"{obj_name}/{remote_save_file}")
            os.remove(load_path)
            os.remove(save_path)
            print(f"[{obj_idx}/{len(obj_list)}] processed: {class_name} - {obj_name}")
            print('============================================================\n')

    t.close()

    print(f'Exit.')
