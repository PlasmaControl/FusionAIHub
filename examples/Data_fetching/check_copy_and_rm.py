"""This script is intended to copy the file from iris and..."""
import paramiko
import paramiko.util
import socket
import getpass
import numpy as np 
import time
import re


num_min = 140000
num_max = 200000-1

subtask = 2  # total paraelle one wants
residue = 1  # transfer num%subtask==residue

fetching_name_list = ['actu', 'basic', 'profiles']
diag_name = fetching_name_list[0]

remote_directory = '/cscratch/curiem/Data_fetch_Basic'
local_directory = '/scratch/gpfs/EKOLEMEN/big_d3d_data/Basic_fetch'

# Set up logging
paramiko.util.log_to_file("paramiko.log")

# SSH settings for the proxy server
proxy_host = 'cybele.gat.com'
proxy_port = 2039  # Example port, modify as necessary
proxy_user = 'curiem'
proxy_password = getpass.getpass(f"Enter SSH password for {proxy_host}: ")

# SSH settings for the destination server
destination_host = 'iris.gat.com'
destination_user = 'curiem'
destination_password = proxy_password


def create_ssh_connection():
    try:
        # Tunneling through the proxy
        proxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy_sock.connect((proxy_host, proxy_port))

        proxy_transport = paramiko.Transport(proxy_sock)
        proxy_transport.connect(username=proxy_user, password=proxy_password)
        proxy_channel = proxy_transport.open_channel(
            kind='direct-tcpip', dest_addr=(destination_host, 22),
            src_addr=(proxy_host, proxy_port))

        # Create an SSH client and connect through the proxy channel
        ssh_client = paramiko.SSHClient()
        # Use with caution in production
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(destination_host, username=destination_user,
                           password=destination_password, sock=proxy_channel)

        return ssh_client, proxy_transport
    except paramiko.AuthenticationException:
        print("Authentication failed, please verify your credentials.")
        return None, None
    except paramiko.SSHException as sshException:
        print(f"Could not establish SSH connection: {sshException}")
        return None, None
    except Exception as e:
        print(f"Exception in connecting to the SSH Server: {e}")
        return None, None


def extract_shot_numbers_remote(sftp, remote_path, suffix):
    """
    Extract shot numbers from file names in a remote directory via SFTP, based
    on a given suffix.

    Parameters
    ----------
    sftp : SFTP object
        An active SFTP client session.
    remote_path : str
        The remote directory path to search for files.
    suffix : str
        The suffix pattern to match in the file names.

    Returns
    -------

    set
        A set of unique shot numbers extracted from the file names.
    """
    try:
        # List all files in the remote directory with their attributes
        files_attr = sftp.listdir_attr(remote_path)

        # Regular expression pattern to match the shot numbers,
        # incorporating the suffix variable
        pattern = re.compile(rf'(\d+)_({suffix})\.h5')

        # Extract shot numbers that match the pattern from the file names
        shot_numbers = {
            int(match.group(1))
            for attr in files_attr
            for match in [pattern.search(attr.filename)] if match
        }

        return shot_numbers
    except Exception as e:
        print(f"Failed to extract shot numbers: {e}")
        return set()

def copy_file(sftp, remote_path, local_path):
    # Check the action and perform the corresponding task
    sftp.get(remote_path, local_path)
    message = f"File {remote_path} copied to {local_path} successfully."
    return message

def remove_file(sftp, remote_path):
    sftp.remove(remote_path)
    message = f"File {remote_path} removed successfully."
    return message

def copy_n_rm_file(sftp, remote_path, local_path):
    message = copy_file(sftp, remote_path, local_path)
    print(message)
    message = remove_file(sftp, remote_path)
    print(message)


def search_copy_and_delete(diag_name, remote_directory, local_directory,
                           retries=3):
    for attempt in range(retries):
        try:
            ssh_client, proxy_transport = create_ssh_connection()

            sftp = ssh_client.open_sftp()
            while 1 == 1:
                shot_numbers = extract_shot_numbers_remote(
                    sftp, remote_directory, diag_name)
                shot_numbers = list(shot_numbers)
                
                shot_numbers.sort()
                shot_numbers = np.array(shot_numbers)
                shot_numbers = shot_numbers[
                    (num_min <= shot_numbers) & (shot_numbers <= num_max)]
                
                # print(shot_numbers)
                if len(shot_numbers) <= 2*subtask:
                    print('No files to copy, waiting for 10 min')
                    # wait for 10min
                    time.sleep(600)
                    continue 
                cp_shot_num = shot_numbers[:-2*subtask]
                print(cp_shot_num)
                for shot_num in cp_shot_num:
                    if int(shot_num) % subtask == residue:
                        for name_tmp in fetching_name_list:
                            remote_path = f'{remote_directory}/{shot_num}_{name_tmp}.h5'
                            local_path = f'{local_directory}/{shot_num}_{name_tmp}.h5'
                            copy_n_rm_file(sftp, remote_path, local_path)

            ssh_client.close()
            proxy_transport.close()
            break
        except Exception as e:
            print(f"Error processing on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(5)
            else:
                print(f"Failed processing after {retries} attempts.")
            ssh_client.close()
            proxy_transport.close()
        finally:
            ssh_client.close()
            proxy_transport.close()
                

search_copy_and_delete(diag_name, remote_directory, local_directory, retries=3)
