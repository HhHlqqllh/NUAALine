function PowerLineExtract() {
	if(global.select_object!=null){
		if(global.select_object.name.endsWith('_converted')){
			let select_filename = global.select_object.name.slice(0, -10)   //获取目标文件名字
			let select_filepath = global.inputpaths.get(global.select_object.name)

			let message = `开始提取电力线<br>
			目标名字: ${select_filename}<br>
			目标路径: ${select_filepath}`;
			viewer.postMessage(message, {duration: 3000});

			progress()
			fetch(`http://localhost:5000/api/save_path?path=${encodeURIComponent(global.inputpaths.get(global.select_object.name))}`)
				.then(response => response.json())
				.then(data => {
					finish_flag = true;   //修改提取结束标志

					let message2 = `电力线提取成功，保存并加载结果!`;
					viewer.postMessage(message2, {duration: 3000});
					
					const lasfiles = [];
					let lasfile = '.\\results\\' + select_filename + '\\' + select_filename + '_results_orgin.las';   //结果文件路径
					lasfiles.push(lasfile)
					let sugdir = '.\\results\\' + select_filename + '\\' + select_filename + '_results_orgin_converted';  //转换路径
					let sugname = select_filename + '_results_orgin_converted';  //树中文件显示名

					convert_17(lasfiles, sugdir, sugname);  //加载结果文件

					const lasfiles2 = [];
					let lasfile2 = '.\\results\\' + select_filename + '\\' + select_filename + '_results_reconstruction.las';   //结果文件路径
					lasfiles2.push(lasfile2)
					let sugdir2 = '.\\results\\' + select_filename + '\\' + select_filename + '_results_reconstruction_converted';  //转换路径
					let sugname2 = select_filename + '_results_reconstruction_converted';  //树中文件显示名

					convert_17(lasfiles2, sugdir2, sugname2);  //加载结果文件

					const lasfiles3 = [];
					let lasfile3 = '.\\results\\' + select_filename + '\\' + select_filename + '_danger.las';   //结果文件路径
					lasfiles3.push(lasfile3)
					let sugdir3 = '.\\results\\' + select_filename + '\\' + select_filename + '_danger_converted';  //转换路径
					let sugname3 = select_filename + '_danger_converted';  //树中文件显示名

					convert_17(lasfiles3, sugdir3, sugname3);  //加载结果文件

					const lasfiles4 = [];
					let lasfile4 = '.\\results\\' + select_filename + '\\' + select_filename + '_towers.las';   //结果文件路径
					lasfiles4.push(lasfile4)
					let sugdir4 = '.\\results\\' + select_filename + '\\' + select_filename + '_towers_converted';  //转换路径
					let sugname4 = select_filename + '_towers_converted';  //树中文件显示名

					convert_17(lasfiles4, sugdir4, sugname4);  //加载结果文件

					global.select_object = null  //清空选中目标
				});
		}
		else{
			let message = `未选择点云！`;
			viewer.postMessage(message, {duration: 15000});
		}
	}
	else{
		let message = `未选择点云!`;
		viewer.postMessage(message, {duration: 15000});
	}
};


function progress() {
	console.log("Connecting to SSE...");
	const progressBar = document.getElementById('progressBar');
	//const progressBarText =  document.getElementById('progressText');
	const source = new EventSource('http://localhost:5000/progress');

	source.onopen = function() {
		console.log('SSE 连接已建立');
	};

	source.onmessage = function(event) {
		console.log('收到进度:', event.data);
		const progressValue = parseFloat(event.data);
		if (!isNaN(progressValue)) {
			console.log('更新进度条:', progressValue);
			//progressBar.value = Math.min(progressValue, 100);
			progressBar.style.width = Math.min(progressValue, 100) + '%';
			progressBar.textContent = Math.min(progressValue, 100) + '%'
			if (progressValue >= 100) {
				//progressBar.value = 0
				progressBar.style.width = 0 + '%';
				progressBar.textContent = 0 + '%'
				source.close();
				console.log('SSE 连接已关闭，进度完成');
			}
		} else {
			console.log('无效进度数据:', event.data);
		}
	};

	source.onerror = function(error) {
		console.error('SSE 错误:', error);
		source.close();
	};
}

function PowerLineExtract2() {
    console.log("PowerLineExtract button clicked!");
    console.log(global.select_object.name.slice(0, -10))
    console.log(global.inputpaths.get(global.select_object.name))
    fetch(`http://localhost:5000/api/save_path?path=${encodeURIComponent(global.inputpaths.get(global.select_object.name))}`)
        .then(response => response.json())
        .then(data => {
			const inputPaths = [];
			let lasfile = './las/temp.las';   //结果文件路径
			inputPaths.push(lasfile)
            //const inputPaths = './las/temp.las';
            const chosenPath = './las/temp_convert';
            const pointcloudName = 'temp_convert';
            console.log(inputPaths,chosenPath)
            convert_17(inputPaths, chosenPath, pointcloudName);
            //loadDroppedPointcloud('./las/temp_convert');
        })
};

function PowerLineExtract3() {
	if(global.select_object!=null){
		if(global.select_object.name.endsWith('_converted')){
			let select_filename = global.select_object.name.slice(0, -10)   //获取目标文件名字
			let select_filepath = global.inputpaths.get(global.select_object.name)

			let message = `Starting extract powerline.<br>
			select_filename: ${select_filename}<br>
			select_filepath: ${select_filepath}`;
			viewer.postMessage(message, {duration: 15000});

			let finish_flag = false;   //提取结束标志
			let start_time = new Date();  //开始时间节点

			const { spawn } = require('child_process');  //调用提取程序
			let exe = './powerlineextract617/powerlineextract617.exe';  //提取程序
			let parameters = [
				select_filepath
			];  //传递文件名字给提取程序
			const converter = spawn(exe, parameters);  //执行提取程序

			converter.on('close', (code) => {
				finish_flag = true;   //修改提取结束标志

				let message2 = `Extracting successfully!<br>
				Loading results.`;
				viewer.postMessage(message2, {duration: 15000});
				
				const lasfiles = [];
				let lasfile = './results/' + select_filename + '/' + select_filename + '_results_orgin.las';   //结果文件路径
				lasfiles.push(lasfile)
				let sugdir = './results/' + select_filename + '/' + select_filename + '_results_orgin_converted/';  //转换路径
				let sugname = select_filename + '_results_orgin_converted';  //树中文件显示名

				convert_17(lasfiles, sugdir, sugname);  //加载结果文件

				const lasfiles2 = [];
				let lasfile2 = './results/' + select_filename + '/' + select_filename + '_results_reconstruction.las';   //结果文件路径
				lasfiles2.push(lasfile2)
				let sugdir2 = './results/' + select_filename + '/' + select_filename + '_results_reconstruction_converted/';  //转换路径
				let sugname2 = select_filename + '_results_reconstruction_converted';  //树中文件显示名

				convert_17(lasfiles2, sugdir2, sugname2);  //加载结果文件

				const lasfiles3 = [];
				let lasfile3 = './results/' + select_filename + '/' + select_filename + '_danger.las';   //结果文件路径
				lasfiles3.push(lasfile3)
				let sugdir3 = './results/' + select_filename + '/' + select_filename + '_danger_converted/';  //转换路径
				let sugname3 = select_filename + '_danger_converted';  //树中文件显示名

				convert_17(lasfiles3, sugdir3, sugname3);  //加载结果文件

				global.select_object = null  //清空选中目标
			});

			const extracting_time = () => {
				if (finish_flag==false) {
					let now_time = new Date();
					let timeDiff = now_time.getTime() - start_time.getTime();
					let message = `Extracting time: ${String(Math.floor((timeDiff % (1000 * 60 * 60)) / (1000 * 60))).padStart(2, '0')}:${String(Math.floor((timeDiff % (1000 * 60)) / 1000)).padStart(2, '0')}`;
					viewer.postMessage(message, {duration: 2000}); 
					setTimeout(extracting_time, 3000); 
				}
			};
			extracting_time();   //计时执行时长
		}
		else{
			let message = `No selected object for powerline extraction!`;
			viewer.postMessage(message, {duration: 15000});
		}
	}
	else{
		let message = `No selected object for powerline extraction!`;
		viewer.postMessage(message, {duration: 15000});
	}
};