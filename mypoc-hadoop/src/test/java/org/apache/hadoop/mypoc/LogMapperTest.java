package org.apache.hadoop.mypoc;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mrunit.mapreduce.MapDriver;
import org.junit.Test;

/**
 * Created by wenjusun on 5/30/16.
 */
public class LogMapperTest {

    @Test
    public void processValidRecord()throws Exception{
        Text value = new Text();


        value.set("2016-02-29 23:59:59,690 +0000 [0:0:0:0:0:0:0:1] INFO  [qtp389572888-27332] com.motorola.blur.cloudsvc.service.CloudService#invoke(579) - [CloudService.Report.API]: api=/v1/checkinuploader/upload appid=YDYWOLQB1NM35HHYPKOZW3V3Z33TC85I userid=null deviceid=423748869321691136 status=200 time=2891 method=POST service=ccs_uploader URI=/v1/checkinuploader/upload.pb querystring:deviceid=423748869321691136&appId=YDYWOLQB1NM35HHYPKOZW3V3Z33TC85I&geolocation=China-East&geocheckintimestamp=1456790396657");

        new MapDriver<LongWritable,Text,Text, IntWritable>()
                .withMapper(new LogMapper())
                .withInput(new LongWritable(1),value)
                .withOutput(new Text("423748869321691136"),new IntWritable(1))
                .runTest();


        value.set("2016-02-29 23:59:59,733 +0000 [0:0:0:0:0:0:0:1] INFO  [qtp389572888-27320] com.motorola.blur.cloudsvc.service.CloudService#handleHttpCall(1884) - [CloudService.Report.APIHandler]: api=/v1/cs/checkin handler=http://lb51-ccs-p-mmi/checkin-cache-service-1.0/ws/checkin-cache-service/2/storeData?appid=6R1TANEX3ZMQ6EU1UH43P34C8B868KTE&_indigo_dc=cn-east&_indigo_zone=CN&_trackid=1456790399710&deviceid=964122792393818112 result=200 time=23");
        new MapDriver<LongWritable,Text,Text, IntWritable>()
                .withMapper(new LogMapper())
                .withInput(new LongWritable(1),value)
                .runTest();


    }
}
