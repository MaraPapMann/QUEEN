import os
import subprocess


def get_subdirs(dir:str):
    '''
    @Args:
        dir: The path to the directory;
    @Desc:
        Get all subdirectories in the directory.
    '''
    return [x[0] for x in os.walk(dir)][1:]


def get_files(dir:str, ext:str) -> list:
    """
    @Desc: To get all files in directory by the extension.
    @Params:
        dir: string, path to the directory;
        ext: string, the extension name;
    @Return:
        files: list of strings, all files in the directory with the extension name.
    """
    files = []
    if dir[-1] != '/':
        dir = dir + '/'
    try:
        files += [dir + each for each in os.listdir(dir) if each.endswith(ext)]
    except Exception as e:
        print(e)
    return files


def rm_rf(pth:str) -> None:
    '''
    @Desc: To recursively remove objects by force.
    @Args:
        pth: Path to the object to remove;
    '''
    os.system('rm -rf %s'%pth)
    return


def find_fp(dir:str, f_name:str) -> list:
    '''
    @Desc: To recursively find all files named after f_name in dir.
    @Args:
        dir: Directory
        f_name: File name
    '''
    command = 'find %s -name %s'%(dir, f_name)
    f_paths = subprocess.getoutput(command)
    f_paths = f_paths.split('\n')
    return f_paths


def pth_exist(pth:str) -> bool:
    '''
    @Desc: To check whether a path exists.
    @Args:
        pth: Path to the object;
    '''
    return os.path.exists(pth)


def join(pth_a:str, pth_b:str) -> str:
    return os.path.join(pth_a, pth_b)


def mkdir(dir:str) -> None:
    paths = dir.split('/')
    dir = ''
    for p in paths:
        dir = join(dir, p)
        if pth_exist(dir):
            pass
        else:
            os.mkdir(dir)
    return


def get_dir(pth:str) -> str:
    '''Get the directory of this path.'''
    return os.path.dirname(pth)


'''Debug'''
if __name__ == '__main__':
    # print(get_subdirs('DATM'))
    print(get_dir('defenses/adversary'))