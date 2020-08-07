for folder in folders:
    for item in folder:
        if item.endswith('.txt'):
		    item.delete
        elif item.endswith('.csv'):
        	station = 'regex the _*_ and ditch _s'
		    new_name = f"{folder}_{station}"
		    shutil.move(item, f"../{new_name}")

