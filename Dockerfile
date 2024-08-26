FROM python:3.11.5

RUN apt update
RUN apt install git -y
RUN apt install gnutls-bin -y
RUN mkdir blender-git
RUN cd blender-git && git clone https://projects.blender.org/blender/blender.git

RUN apt-get update && apt-get -y install sudo

RUN cd blender-git/blender/ && ./build_files/build_environment/install_linux_packages.py

RUN mkdir blender-git/lib
RUN cd blender-git/lib && svn checkout https://svn.blender.org/svnroot/bf-blender/trunk/lib/linux_x86_64_glibc_228

RUN cd blender-git/blender && make update

RUN cd blender-git/blender && make bpy
RUN cd blender-git/blender && make

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs

RUN ls blender-git

RUN git clone https://github.com/mortorit/ShapeNetRenderer.git
RUN cp -r blender-git/build_linux_bpy/bin/bpy ShapeNetRenderer/
RUN pip install open_clip_torch torch torchvision torchaudio
RUN pip install scipy
RUN pip install trimesh torch