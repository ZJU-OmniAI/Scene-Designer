"""Apply the minimal integration patch to a separately obtained SceneSmith tree."""
import argparse
from pathlib import Path
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--path',type=Path,default=Path('scenesmith'))
    p.add_argument('--check',action='store_true',help='Validate patch applicability without changing files')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    target=args.path.resolve()
    expected=(root/'integration/scenesmith-base.txt').read_text().strip()
    head=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
    if head!=expected:raise SystemExit(f'Expected SceneSmith commit {expected}; found {head}')
    patch=root/'integration/scenesmith.patch'
    cmd=['git','-C',str(target),'apply']
    check=subprocess.run(cmd+['--check',str(patch)],capture_output=True,text=True)
    if check.returncode:
        reverse=subprocess.run(cmd+['--reverse','--check',str(patch)],capture_output=True,text=True)
        if reverse.returncode:raise SystemExit('Patch does not apply cleanly. Use a fresh checkout.\n'+check.stderr)
        print('Integration patch already applied.')
    elif not args.check:
        subprocess.run(cmd+[str(patch)],check=True)
    if not args.check:
        shutil.copyfile(root/'integration/scene_designer.yaml',target/'configurations/scene_designer.yaml')
    print('SceneSmith integration check passed.' if args.check else 'SceneSmith integration ready.')
if __name__=='__main__':main()
