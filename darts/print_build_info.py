import os

def print_build_info():
    here = os.path.abspath(os.path.dirname(__file__))
    version_info_file = os.path.join(here, 'build_info.txt')
    if os.path.exists(version_info_file):
        with open(version_info_file, 'r') as fp:
            date_time = fp.readline().rstrip()
            user_host = fp.readline().rstrip()
            git_hash = fp.readline().rstrip()
            print ('darts-package built on %s by %s from %s' % (date_time, user_host, git_hash))
    else:
        import subprocess
        try:
            git_hash = subprocess.run(['git', 'describe', '--always', '--dirty'], stdout=subprocess.PIPE, cwd=here)
        except FileNotFoundError:
            print('darts-package is run locally from %s [no git hash info available]', here)
            return
        print('darts-package is imported locally from %s [%s]' % (here, git_hash.stdout.decode('utf-8').rstrip()))
