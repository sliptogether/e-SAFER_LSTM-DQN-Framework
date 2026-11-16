import argparse
import _bootstrap
from sweat_rl.config import load_config, project_path
from sweat_rl.data import preprocess_day
from sweat_rl.utils import ensure_dir

def main(config):
    cfg=load_config(config)
    raw=project_path(cfg,cfg["paths"]["raw_dir"])
    out=ensure_dir(project_path(cfg,cfg["paths"]["output_dir"],"processed"))
    for d in [1,2,3,4]:
        key=f"Day{d}"
        path=raw/f"{key}.xlsx"
        if not path.exists():
            raise FileNotFoundError(path)
        df=preprocess_day(path,key,cfg)
        df.to_csv(out/f"{key}.csv",index=False)
        print(key,len(df))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/denoised_5min.yaml")
    main(p.parse_args().config)
