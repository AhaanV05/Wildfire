#!/usr/bin/env python3
"""
Parallel Dataset Builder Launcher

Runs multiple instances of the dataset builder in parallel to speed up processing.
Each instance processes a different subset of the ros_targets table.
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path

def run_parallel_instances(total_instances=4, limit=None, other_args=""):
    """
    Launch multiple parallel instances of the dataset builder.
    
    Args:
        total_instances: Number of parallel processes to run
        limit: Total limit of records to process (split across instances)
        other_args: Additional arguments to pass to each instance
    """
    
    # Get absolute paths relative to this script's location
    script_dir = Path(__file__).parent.absolute()
    python_exe = script_dir.parent / "threat" / "Scripts" / "python.exe"
    script_path = script_dir / "dataset_build_feature.py"
    
    # Verify paths exist
    if not python_exe.exists():
        print(f"❌ Python executable not found: {python_exe}")
        return
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return
    
    processes = []
    
    print(f"🚀 Starting {total_instances} parallel instances...")
    print(f"📊 Total records to process: {limit if limit else 'All remaining'}")
    print(f"⚡ Each instance will process ~{limit // total_instances if limit else 'N/A'} records")
    print("-" * 60)
    
    # Start all instances
    for i in range(total_instances):
        cmd = [
            str(python_exe),
            str(script_path),
            "--instance", str(i),
            "--total-instances", str(total_instances)
        ]
        
        if limit:
            cmd.extend(["--limit", str(limit)])
            
        if other_args:
            cmd.extend(other_args.split())
        
        print(f"🏃 Starting Instance {i + 1}: {' '.join(cmd)}")
        
        # Start process from the correct working directory
        process = subprocess.Popen(
            cmd,
            cwd=str(script_dir),  # Set working directory to Dataset_Builder
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        processes.append({
            'process': process,
            'instance': i + 1,
            'cmd': ' '.join(cmd)
        })
        
        # Small delay between starts
        time.sleep(2)
    
    print("-" * 60)
    print("✅ All instances started! Monitoring progress...")
    print("💡 Press Ctrl+C to stop all instances")
    print("-" * 60)
    
    try:
        # Monitor all processes
        while any(p['process'].poll() is None for p in processes):
            for p in processes:
                if p['process'].poll() is None:  # Still running
                    # Read output line by line
                    while True:
                        output = p['process'].stdout.readline()
                        if output:
                            print(f"[Instance {p['instance']}] {output.strip()}")
                        else:
                            break
                else:
                    # Process finished
                    return_code = p['process'].returncode
                    if return_code == 0:
                        print(f"✅ Instance {p['instance']} completed successfully!")
                    else:
                        print(f"❌ Instance {p['instance']} failed with code {return_code}")
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping all instances...")
        for p in processes:
            if p['process'].poll() is None:
                p['process'].terminate()
                print(f"🛑 Stopped Instance {p['instance']}")
        
        # Wait for all to terminate
        for p in processes:
            p['process'].wait()
    
    print("\n🏁 All instances finished!")

def main():
    parser = argparse.ArgumentParser(description="Run multiple parallel dataset builder instances")
    parser.add_argument("--instances", type=int, default=4, 
                       help="Number of parallel instances (default: 4)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Total number of records to process across all instances")
    parser.add_argument("--other-args", default="",
                       help="Additional arguments to pass to each instance (in quotes)")
    
    args = parser.parse_args()
    
    if args.instances < 1:
        print("❌ Number of instances must be at least 1")
        sys.exit(1)
    
    if args.instances > 8:
        print("⚠️  Warning: Running more than 8 instances may overwhelm your system")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    run_parallel_instances(args.instances, args.limit, args.other_args)

if __name__ == "__main__":
    main()