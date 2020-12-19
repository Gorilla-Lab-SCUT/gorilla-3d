# Scripts for generating data for `shapenet_implicit_recon` dataset

1. prepare the necessary files

    first, please download files:
    - download [shapenet v1](http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1.zip) dataset
    - download [rendered rgb images](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz)
    - download [sdf executable](https://github.com/laughtervv/DISN/raw/master/isosurface/computeDistanceField) (note: to run computeDistanceField successfully, you may need libtbb.so.2 / libtbb_preview.so.2 / libtcmalloc.so.4 (please specify LD_LIBRARY_PATH manually))
    - download [meshlabserver](https://github.com/cnr-isti-vclab/meshlab/releases), either linux (suggested) or windows will do. Ideally, meshlabserver should be installed on the platform you run the codes (e.g. linux); but it can also be installed on another machine to run meshlabserver but does not have the data (e.g. windows).

    then, install some requirements:
    ```bash
    pip install pymcubes
    pip install paramiko
    conda install -c conda-forge point_cloud_utils
    ```

    finally, unzip those files to some locations.
    assuming shapenet is unzipped to `$ShapeNetFolder`, images is unzipped to `$ImagesFolder`, the path to sdf executable is `$SDFexe`,
    and the directory `$SaveFolder` is where you want to save the results (make sure you have enough disk space)

2. run scripts

    since [3D-R2N2](https://github.com/chrischoy/3D-R2N2) only provides 13 categories, we can generate up to 13 categories.
    they are:

        - 02691156
        - 02828884
        - 02933112  
        - 02958343  
        - 03001627
        - 03211117  
        - 03636649  
        - 03691459  
        - 04090263  
        - 04256520  
        - 04379243  
        - 04401088  
        - 04530566
    
    we choose two categories: `03001627` and `02691156` for demonstration.
    assuming you are under the directory `shapenet_implicit_recon`, please run the following commands in order.

    ```bash
    # categories to process
    export categories="03001627 02691156"
    
    ##########################################
    # generate split file
    python -m gorilla3d.preprocessing.shapenet.shapenet_implicit_recon.0_get_split.py --class_name $categories --src_dataset_dir $ShapeNetFolder --split_dir $SaveFolder/original_split
    
    ##########################################
    # create meshes
    for sp in train val test; do
        python -m gorilla3d.preprocessing.shapenet.shapenet_implicit_recon.1_create_mesh.py --class_name $categories --split $sp --src_dataset_dir $ShapeNetFolder \
                                --dst_dataset_dir $SaveFolder --split_dir $SaveFolder/original_split --sdf_executable $SDFexe
    done

    ##########################################
    # simplify mesh
    # if your meshlabserver is installed on the machine storing your dataset, please run this single command
    python -m gorilla3d.preprocessing.shapenet.shapenet_implicit_recon.2_simplify_mesh.py --class_name $categories --dst_dataset_dir $SaveFolder
    # if your meshlabserver is installed on another machine, but it can login to the machine storing the dataset, please run the following five commands (you may need to specify password by adding an addition argument `--remote_password ???`)
    # if you are on linux:
    export remote_ip="111.222.333.444" # this is your remote machine ip address which has the data
    export remote_user="root" # this is the remote machine user name to login
    export temp_path="." # this is the directory on your local machine which has enough disk space to store the temporary results
    export script="./2_simplify_mesh.mlx" # this is the path to the meshlab script to simplify mesh on local machine
    python -m gorilla3d.preprocessing.shapenet.shapenet_implicit_recon.2_simplify_mesh.py --class_name $categories --dst_dataset_dir $SaveFolder --remote_process \
                              --remote_server $remote_ip --remote_username $remote_user --remote_localpath $temp_path --remote_script_path $script
    # if you are on windows (save them in a .bat script file and run it)
    @echo off
    set categories=03001627 02691156
    set SaveFolder=????
    set remote_ip=???
    set remote_user=???
    set temp_path=???
    set script=???
    set password=???
    python -m gorilla3d.preprocessing.shapenet.shapenet_implicit_recon.2_simplify_mesh.py --class_name %categories% --dst_dataset_dir %SaveFolder% --remote_process --remote_server %remote_ip% --remote_username %remote_user% --remote_localpath %temp_path% --remote_script_path %script% --remote_password %password%
    pause
    
    ##########################################
    # filter out unqualified meshes (note: for '04090263', you may need to specify "--minMB 1.0" additionally)
    mkdir $SaveFolder/unqualified $SaveFolder/split
    for sp in train val test; do
        python -m gorilla3d.preprocessing.shapenet.shapenet_implicit_recon.3_filter_out.py --class_name $categories --split $sp --src_dataset_dir $SaveFolder --moved_dataset_dir $SaveFolder/unqualified \
                               --load_split_dir $SaveFolder/original_split --save_split_dir $SaveFolder/split
    done

    ##########################################
    # generate h5 file finally 
    for sp in train val test; do
        python -m gorilla3d.preprocessing.shapenet.shapenet_implicit_recon.4_gen_h5.py --class_name $categories --split $sp --src_dataset_dir $SaveFolder --dst_dataset_dir $SaveFolder \
                           --img_dataset_dir $ImagesFolder --split_dir $SaveFolder/split 
    done

    ##########################################
    # link
    ln -s $SaveFolder data

    ```

    dataset will load h5 file under the folder `$SaveFolder`

3. issue

    please contact me: lei.jiabao
    