# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import paramiko
import argparse

default_mlx_path = os.path.join(os.path.dirname(__file__), "2_simplify_mesh.mlx")
load_file = "mesh_gt.ply"
save_file = "mesh_gt_simplified.ply"


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

    def remote_ls(self, dir_path="."):
        return self.sftp.listdir(dir_path)

    def remote_exists(self, dir_path, file_name):
        return (file_name in self.remote_ls(dir_path))

    def close(self):
        self.sftp.close()
        self.ssh.close()


def local_process(load_path, save_path, script_path):
    os.system("meshlabserver -i {} -o {} -s {}".format(load_path, save_path, script_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", type=str, nargs="+", default=["03001627"], help="Categories to process")
    parser.add_argument("--dst_dataset_dir",
                        type=str,
                        help="Path to the folder to those saved results used in the previous script")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--remote_process", action="store_true")
    parser.add_argument("--remote_server", type=str, default="", help="Remote server ip address")
    parser.add_argument("--remote_username", type=str, default="", help="Remote server username")
    parser.add_argument("--remote_localpath", type=str, help="Local directory to process")
    parser.add_argument("--remote_script_path",
                        type=str,
                        default=default_mlx_path,
                        help="Path to the meshlab simplification script")
    args = parser.parse_args()

    if len(args.class_name) == 1:

        class_name = args.class_name[0]

        override = args.override
        remote_process = args.remote_process
        script_path = args.remote_script_path

        if remote_process:
            # files are on server, but meshlabserver is on local pc
            # you should run this file on your local pc
            # and .mlx file should also copy to your local pc
            # it will automatically fetch files from remote server, process it, and finally upload it

            server = args.remote_server
            username = args.remote_username
            localpath = args.remote_localpath
            remote_path = args.dst_dataset_dir

            t = Transfer(server, username)
            t.remote_cd(remote_path)
            class_list = t.remote_ls()
            assert class_name in class_list
            obj_list = t.remote_ls(class_name)

            t.remote_cd(class_name)
            for obj_idx, obj_name in enumerate(obj_list):
                if not t.remote_exists(obj_name, save_file) or override:
                    load_path = os.path.join(localpath, "{}_{}_load.ply".format(class_name, obj_name))
                    save_path = os.path.join(localpath, "{}_{}_save.ply".format(class_name, obj_name))
                    t.from_remote("{}/{}".format(obj_name, load_file), load_path)
                    local_process(load_path, save_path, script_path)
                    t.to_remote(save_path, "{}/{}".format(obj_name, save_file))
                    os.remove(load_path)
                    os.remove(save_path)
                    print("[{}/{}] processed: {} - {}".format(obj_idx, len(obj_list), class_name, obj_name))
                    print("============================================================\n")
                else:
                    print("Skip process {} - {}".format(class_name, obj_name))
                    print("============================================================\n")
            t.close()

        else:
            # local process
            load_path = args.dst_dataset_dir

            class_list = os.listdir(load_path)
            assert class_name in class_list
            obj_list = os.listdir(os.path.join(load_path, class_name))
            obj_list = [s for s in obj_list if os.path.isdir(os.path.join(load_path, class_name, s))]

            for obj_idx, obj_name in enumerate(obj_list):
                load_path = os.path.join(load_path, class_name, obj_name, load_file)
                save_path = os.path.join(load_path, class_name, obj_name, save_file)
                if not os.path.exists(save_path) or override:
                    local_process(load_path, save_path, script_path)
                    print("[{}/{}] processed: {} - {}".format(obj_idx, len(obj_list), class_name, obj_name))
                    print("============================================================\n")
                else:
                    print("Skip process {} - {}".format(class_name, obj_name))
                    print("============================================================\n")

    else:
        for class_name in args.class_name:
            os.system(
                "python {} --class_name {} --localpath {} --dst_dataset_dir {} --remote_server {} --remote_username {} --remote_script_path {}"
                .format(
                    __file__,
                    class_name,
                    args.localpath,
                    args.dst_dataset_dir,
                    args.remote_server,
                    args.remote_username,
                    args.remote_script_path,
                ) + (" --override" if args.override else "") + \
                    (" --remote_process" if args.remote_process else ""))

    print("All done.")
