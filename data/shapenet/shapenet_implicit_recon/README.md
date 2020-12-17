# Scripts for generating data for `shapenet_implicit_recon` dataset

1. prepare the necessary files

    first, please download files:
    - download [shapenet v1](http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1.zip) dataset
    - download [rendered rgb images](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz)
    - download [sdf executable](https://github.com/laughtervv/DISN/raw/master/isosurface/computeDistanceField)
    - download [meshlabserver](https://github.com/cnr-isti-vclab/meshlab/releases), either linux (suggested) or windows will do. Ideally, meshlabserver should be installed on the platform you run the codes (e.g. linux); but it can also be installed on another machine to run meshlabserver but does not have the data (e.g. windows).

    then, unzip those files to some locations.
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

    ```bash
    # categories to process
    export categories="03001627 02691156"
    
    ##########################################
    # generate split file
    python 0_get_split.py --class_name $categories --src_dataset_dir $ShapeNetFolder --split_dir $SaveFolder/original_split
    
    ##########################################
    # create meshes
    for sp in train val test; do
        python 1_create_mesh.py --class_name $categories --split $sp --src_dataset_dir $ShapeNetFolder \
                                --dst_dataset_dir $SaveFolder --split_dir $SaveFolder/original_split --sdf_executable $SDFexe
    done

    ##########################################
    # simplify mesh
    # if your meshlabserver is installed on the machine storing your dataset, please run this single command
    python 2_simplify_mesh.py --class_name $categories --dst_dataset_dir $SaveFolder
    # if your meshlabserver is installed on another machine, but it can login to the machine storing the dataset, please run the following five commands
    export remote_ip="111.222.333.444" # this is your remote machine ip address which has the data
    export remote_user="root" # this is the remote machine user name to login
    export temp_path="." # this is the directory on your local machine which has enough disk space to store the temporary results
    export script="./2_simplify_mesh.mlx" # this is the path to the meshlab script to simplify mesh on local machine
    python 2_simplify_mesh.py --class_name $categories --dst_dataset_dir $SaveFolder --remote_process \
                              --remote_server $remote_ip --remote_username $remote_user --remote_localpath $temp_path --remote_script_path $script
    
    ##########################################
    # filter out unqualified meshes (note: for '04090263', you may need to specify "--minMB 1.0" additionally)
    mkdir $SaveFolder/unqualified $SaveFolder/split
    for sp in train val test; do
        python 3_filter_out.py --class_name $categories --split $sp --src_dataset_dir $SaveFolder --moved_dataset_dir $SaveFolder/unqualified \
                               --load_split_dir $SaveFolder/original_split --save_split_dir $SaveFolder/split
    done

    ##########################################
    # generate h5 file finally 
    for sp in train val test; do
        python 4_gen_h5.py --class_name $categories --split $sp --src_dataset_dir $SaveFolder --dst_dataset_dir $SaveFolder \
                           --img_dataset_dir $ImagesFolder --split_dir $SaveFolder/split 
    done

    ##########################################
    # link
    ln -s $SaveFolder data

    ```

    dataset will load h5 file under the folder `$SaveFolder`

3. issue

    author: lei.jiabao
    
    
    - if you meet that `no module named mcubes.`, you need to install mcubes like this:
    
    '''
        pip install pymcubes
    ''' 
    
    -  to run computeDistanceField, you may need libtbb.so.2 / libtbb_preview.so.2 / libtcmalloc.so.4 (specify LD_LIBRARY_PATH manually)
    
    
    - if you meet that `no module named paramiko`, you need to install paramiko like this:
    
    '''
        pip install paramiko
    '''
    
    - if you meet that `no module named point_cloud_util`, you need to install point_cloud_util like this:
    
    '''
        pip install git+git://github.com/fwilliams/point-cloud-utils
    '''
    
    or
    
    '''
        conda install -c conda-forge point_cloud_utils=0.15.1
    '''
