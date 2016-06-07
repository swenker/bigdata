package org.apache.hadoop.mypoc;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * Created by wenjusun on 5/26/16.
 */
public class LogMapper extends Mapper<LongWritable,Text,Text,IntWritable>{

    enum NullField{
        DEVICE,
        API
    }
    IntWritable one = new IntWritable(1);
    Text device = new Text();
    //2016-02-29 23:59:59,690 +0000 [0:0:0:0:0:0:0:1] INFO  [qtp389572888-27332] com.motorola.blur.cloudsvc.service.CloudService#invoke(579) -
    // [CloudService.Report.API]: api=/v1/checkinuploader/upload appid=YDYWOLQB1NM35HHYPKOZW3V3Z33TC85I
    // userid=null deviceid=423748869321691136 status=200 time=2891 method=POST service=ccs_uploader
    // URI=/v1/checkinuploader/upload.pb querystring:deviceid=423748869321691136&appId=YDYWOLQB1NM35HHYPKOZW3V3Z33TC85I&geolocation=China-East&geocheckintimestamp=1456790396657
    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

        String line = value.toString();
//        if (line.contains("[CloudService.Report.API]")) {
        if (line.contains("CloudService#invoke(579)")) {
            String fields[] = line.split(" ");
/*
            int i=0;
            for(String s:fields){
                System.out.print("i="+i++);
                System.out.println("    "+s);
            }
*/

            String apiPart = fields[10];
            String deviceid = fields[13];

            if(deviceid!=null&&deviceid.startsWith("deviceid=")) {
                String idpair[] = deviceid.split("=");
                if(idpair.length==2) {
                    if(idpair[1]==null|| idpair[1].equals("null"))
                        context.getCounter(NullField.DEVICE).increment(1);
                    else {
                        device.set(idpair[1]);
                        context.write(device, one);
                    }
                }
            }

        }

    }
}
