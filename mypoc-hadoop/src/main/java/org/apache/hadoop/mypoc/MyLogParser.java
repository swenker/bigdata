package org.apache.hadoop.mypoc;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;

/**
 * Created by wenjusun on 5/26/16.
 */
/*
public class MyLogParser {
    public static void main(String args[]){
        Configuration conf = new Configuration();

        try {
            Job job = Job.getInstance();
            job.setJobName("MyLogParser");
            job.setJarByClass(MyLogParser.class);
            job.setMapperClass(LogMapper.class);
            job.setReducerClass(LogReducer.class);

            FileInputFormat.addInputPath(job,new Path("/data/cloud-service-1.0.log.2016-02-29-23"));
            FileOutputFormat.setOutputPath(job,new Path("/home/hadoop/mypoc"));

            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(IntWritable.class);

            System.exit(job.waitForCompletion(true)?0:1);

        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}
*/
public class MyLogParser extends Configured implements Tool{

    @Override
    public int run(String[] strings) throws Exception {
        Configuration conf = new Configuration();

        Job job = Job.getInstance(conf);

        job.setJobName("MyLogParser");
        job.setJarByClass(MyLogParser.class);
        job.setMapperClass(LogMapper.class);
        job.setReducerClass(LogReducer.class);

        FileSystem fs = FileSystem.get(conf);

        //Path inputPath = new Path("/data/cloud-service-1.0.log.2016-02-29-23");

        Path outputPath = new Path("/home/hadoop/mypoc");
        fs.delete(outputPath,true);

        System.out.println("Output path cleared.......");
        //FileInputFormat.addInputPath(job,inputPath);
        String dataFiles=strings[0];

        System.out.println(dataFiles);
        FileInputFormat.addInputPaths(job,dataFiles);
        FileOutputFormat.setOutputPath(job,outputPath);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        return job.waitForCompletion(true)?0:1;
//        return 0;

    }

    public static void main(String args[]){

        try {

            int exitCode = ToolRunner.run(new MyLogParser(),args);
            System.exit(exitCode);

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
