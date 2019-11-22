

def get_deleted_files():

    apollo_delete_files = []
    with open('Mesh_overlay_train_error_delete.txt', 'r') as f:
        for line in f.readlines():
            apollo_delete_files.append(line.strip('\n')+'.jpg')

    with open('Mesh_overlay_val_error_delete.txt', 'r') as f:
        for line in f.readlines():
            apollo_delete_files.append(line.strip('\n')+'.jpg')

    pku_delete_files = []
    with open('train_map.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            pku_file, apollo_file = line.split()
            if apollo_file in apollo_delete_files:
                # print(pku_file, '要被删除')
                pku_delete_files.append(pku_file)

    return pku_delete_files


if __name__ == '__main__':
    get_deleted_files()