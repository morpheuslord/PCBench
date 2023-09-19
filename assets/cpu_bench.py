import random
import time
import threading
import math
import platform
import os
import assets.cpu_info as cpu_info
from rich import print
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich.align import Align
from rich.markdown import Markdown
from rich import box

num_samples = 100000
num_iterations = 10
benchmark_duration_fps = 20
num_threads = 4
benchmark_duration = 300
computations_per_second = 1000000

os_verson = platform.system()


def clearscr() -> None:
    try:
        osp = platform.system()
        match osp:
            case 'Darwin':
                os.system("clear")
            case 'Linux':
                os.system("clear")
            case 'Windows':
                os.system("cls")
    except Exception:
        pass


def monte_carlo_pi(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x**2 + y**2 <= 1:
            inside_circle += 1
    return (inside_circle / num_samples) * 4


def benchmark_pi_calculation(num_samples, num_iterations):
    total_time = 0
    for _ in range(num_iterations):
        start_time = time.time()
        pi_estimate = monte_carlo_pi(num_samples)
        end_time = time.time()
        total_time += end_time - start_time
        print(
            f"Iteration {_:2}: π ≈ {pi_estimate:.8f} (Time: {end_time - start_time:.4f} seconds)")

    average_time = total_time / num_iterations
    print(
        f"Average Time for {num_iterations} iterations: {average_time:.4f} seconds")


def calculate_performance(duration, computations):
    total_computations = computations * (duration / 1.0)
    performance = total_computations / duration
    return performance


def cpu_intensive_task(duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        pass


def calculate_fps(frames, duration):
    fps = frames / duration
    return fps


def multi_thread_benchmark(duration, num_threads):
    print(Panel("Single-threaded FPS Benchmark"))
    frames = 0
    start_time = time.time()
    while time.time() - start_time < duration:
        cpu_intensive_task(0.01)
        frames += 1
    single_fps = calculate_fps(frames, duration)
    print(f"Average FPS: {single_fps:.2f}")

    print(Panel("Multi-threaded FPS Benchmark"))
    frames = 0
    start_time = time.time()
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(
            target=cpu_intensive_task, args=(duration/num_threads,))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
        frames += int(duration/num_threads)
    multi_fps = calculate_fps(frames, duration)
    print(f"Average FPS: {multi_fps:.2f}")


def bench(num_samples, num_iterations):
    clearscr()
    options = f"""
        CPU: {cpu_info.get_cpu_info().get('brand_raw', "Unknown")}
        ARCH: {cpu_info.get_cpu_info().get('arch_string_raw', "Unknown")}
        OS: {str(os_verson)}

        Num Samples: {num_samples}
        Num Iterations: {num_iterations}
    """
    opt = Markdown(options)
    option_panel = Panel(
        Align.center(
            Group("\n", Align.center(opt)),
            vertical="middle",
        ),
        box=box.ROUNDED,
        padding=(1, 2),
        title="[b red]Bench Options",
        border_style="blue",
    )
    print(option_panel)
    print(
        f"Calculating π using Monte Carlo with {num_samples} samples...\n")
    benchmark_pi_calculation(num_samples, num_iterations)


def pi_menu():
    global num_samples
    global num_iterations
    pi_menu_table = Table()
    pi_menu_table.add_column("Option")
    pi_menu_table.add_column("Value")
    pi_menu_table.add_column("DefaultValue")
    pi_menu_table.add_row("1", "Change Num Samples", str(num_samples))
    pi_menu_table.add_row("2", "Change Num Iterations", str(num_iterations))
    pi_menu_table.add_row("3", "Change Both", "Your Customs")
    pi_menu_table.add_row("4", "Defaults", f"{num_samples}, {num_iterations}")
    print(pi_menu_table)
    opt = input("Choose an option: ")
    match opt:
        case "1":
            num_samples = input("Enter the Num Samples:")
            bench(int(num_samples), int(num_iterations))
        case "2":
            num_iterations = input("Enter the Num Iterations:")
            bench(int(num_samples), int(num_iterations))
        case "3":
            num_samples = input("Enter the Num Samples:")
            num_iterations = input("Enter the Num Iterations:")
            bench(int(num_samples), int(num_iterations))
        case "4":
            bench(int(num_samples), int(num_iterations))


def fps_menu():
    global benchmark_duration_fps
    global num_threads
    fps_menu_table = Table()
    fps_menu_table.add_column("Option")
    fps_menu_table.add_column("Value")
    fps_menu_table.add_column("DefaultValue")
    fps_menu_table.add_row(
        "1", "Change Benchmark Duration", str(benchmark_duration_fps))
    fps_menu_table.add_row("2", "Change Num of threads", str(num_threads))
    fps_menu_table.add_row("3", "Change Both", "Your Customs")
    fps_menu_table.add_row(
        "4", "Defaults", f"{benchmark_duration_fps}, {num_threads}")
    print(fps_menu_table)
    opt = input("Choose an option: ")
    match opt:
        case "1":
            benchmark_duration_fps = input("Enter the Benchmark Duration:")
            multi_thread_benchmark(
                int(benchmark_duration_fps), int(num_threads))
        case "2":
            num_threads = input("Enter the Num of threads:")
            multi_thread_benchmark(
                int(benchmark_duration_fps), int(num_threads))
        case "3":
            benchmark_duration_fps = input("Enter the Benchmark Duration:")
            num_threads = input("Enter the Num of threads:")
            multi_thread_benchmark(
                int(benchmark_duration_fps), int(num_threads))
        case "4":
            multi_thread_benchmark(
                int(benchmark_duration_fps), int(num_threads))


def cpu_bench():
    clearscr()
    cpu_bench_banner = """
     _______  _______  __   __  _______  _______  __    _  _______  __   __
    |       ||       ||  | |  ||  _    ||       ||  |  | ||       ||  | |  |
    |       ||    _  ||  | |  || |_|   ||    ___||   |_| ||       ||  |_|  |
    |       ||   |_| ||  |_|  ||       ||   |___ |       ||       ||       |
    |      _||    ___||       ||  _   | |    ___||  _    ||      _||       |
    |     |_ |   |    |       || |_|   ||   |___ | | |   ||     |_ |   _   |
    |_______||___|    |_______||_______||_______||_|  |__||_______||__| |__|
    """
    cpu_banner = Markdown(cpu_bench_banner)
    print(cpu_banner)
    cpu_menu = Table()
    cpu_menu.add_column("Option")
    cpu_menu.add_column("Benchmark")
    cpu_menu.add_row("1", "PI Calculations")
    cpu_menu.add_row("2", "FPS Benchmark")
    cpu_menu.add_row("3", "Quit")
    print(cpu_menu)
    opt = input("Choose an option: ")
    match opt:
        case "1":
            clearscr()
            print(cpu_banner)
            pi_menu()
        case "2":
            clearscr()
            print(cpu_banner)
            fps_menu()
        case "3":
            quit()
