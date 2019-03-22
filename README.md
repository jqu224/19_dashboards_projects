# 19_intern_projects

----------------------------
Project one:
fetch error message and generate weekly summary
----------------------------

![alt text](https://github.com/jqu224/19_intern_projects/blob/master/image/30/Untitled%20Diagram-Page-2%20(2).png "Flowchart")
```

requirements:
	two series of [slice] products:
				a: consists of 10 [items]
				b: consists of either 10 items or 40 [items]
	for each week the team is expecting ~100 slices to be logged

	for each slice: 
		1 - go to folder of [slice] maps
			fetch the error msg according to the error code from the excel map
		2 - go to individual folder of each [slice] 
			display a - csv log for voltage - Open and Short Circuit info
			        b -  Microscopic Images - physical inspections 
	after checking on the error msg and inspection images respectively gather the error code and output it to a spreadsheet

options:
	software: JMP with jmp scripting language / Python with Jupyter NoteBook 
	folders: 1 - slice maps [*map]
		 2 - single slice folder [*fold] consists of images and csv logs 
	

Solution
	step 0: prepare a list of [slice ids] in csv files without header
		id : type : [other_columns]	
		for instance: 1234, K_101 or 2234, W_88
	      	such infomation can be gathered from weekly production file 			[slice production.xls]
	step 1: OPEN a display window upon triggering the script
	   	ask the user to input a list of csv file and let the user to choose from the list 	[radio button]
	   	alter: ask the user to type in a single slice id and it's product type 			[text edit box]
	step 2: go to the [*map] for the slice map and 
		generate a [n_items by 5] table in the display window by decoding the original_err_msg, 
		5 stands for 5 columns: original_err_msg : 
					location_code : 
					location_msg :
					error_code : 
					detailed_error_msg 
	step 3: list image names by [item number] 
		and let the user to choose which image to show for QA purpose				[radio button]
	step 4: let user to update 40 and 20 counters for the electrical/mechanical errors respectively	[spin box] 
	step x: now the user should be able to select and paste the entire table to other spreadsheets using ctrl+c/v

detailed breakdown: 
/////////////////////////////////////////////////////////////////////////////
/////////////////////////// README.JSL ////////////////////////////////////// 
////////// Here is a brief intro for the usage of this script  //////////////
////// A,	enter the target [SLICE ID] and [TYPE]		//////////////// 
////// B,	click on [SHOW] 				//////////////// 
//////		and the script would pull out a table of [ERR INFO] 	/////////
//////		stored under the location "S:\xxx Data\Slice Maps\"	/////////
////// C,	in that table of [ERR INFO] from step b, there is a list //////// 
//////  	of [radio buttons] (10 or 31 items), click on one of them  ///// 
////// D,	on the right hand side, there are two buttons 		/////////////
//////  	one is to open up the [IMAGE FOLDER] based on the [SLICE ID] ////
////// 	    	the other is to [LIST IMAGE] by the [ITEM NUMBER] from step C ////
////// E,	click on [SHOW THE SELECTED IMAGE] and image from 	///////// 
////// 		"S:\_product_type_\_slice_id_" will be shown below	/////////
////// F,	update the row of error by clicking on 			/////////
////// 		40 probe and 20 mechanical [Spin Boxes] 		///////// 
////// 		then click on update the table to output the data to the table //
//////////////////////////////  END of README  //////////////////////////////
/////////////////////////////////////////////////////////////////////////////
	
		
```

