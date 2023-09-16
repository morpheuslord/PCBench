import assets.cpu_bench as cpu_bench
import assets.gpu as gpu
from rich.markdown import Markdown
from rich import print
from rich.table import Table

if __name__ == "__main__":
    main_banner = """
     _______  _______  _______  _______  __    _  _______  __   __
    |       ||       ||  _    ||       ||  |  | ||       ||  | |  |
    |    _  ||       || |_|   ||    ___||   |_| ||       ||  |_|  |
    |   |_| ||       ||       ||   |___ |       ||       ||       |
    |    ___||      _||  _   | |    ___||  _    ||      _||       |
    |   |    |     |_ | |_|   ||   |___ | | |   ||     |_ |   _   |
    |___|    |_______||_______||_______||_|  |__||_______||__| |__|
    By: Morpheuslord
    """
    banner = Markdown(main_banner)
    print(banner)
    main_table = Table()
    main_table.add_column("options")
    main_table.add_column("Name")
    main_table.add_row("1", "CPU Bench")
    main_table.add_row("2", "GPU Bench")
    print(main_table)
    opt = input("Enter options:")
    match opt:
        case "1":
            cpu_bench.cpu_bench()
        case "2":
            gpu.gpu_bench_menu()
