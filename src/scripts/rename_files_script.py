'''
Created on Nov 26, 2018

@author: root
'''
import os

def rename_files(dir):
    for file_name in os.listdir(dir):
        old_file = os.path.join(dir, file_name)
        if "seg" in file_name:
            fn_split = file_name.split("_")
            new_name = ""
            for i in range(len(fn_split)-1):
                new_name += fn_split[i]+"_"
            new_name += "html_"+fn_split[-1]
            new_file = os.path.join(dir, new_name)
        else:
            new_file = os.path.join(dir, file_name.replace("_html.txt", ".txt"))
        #os.rename(old_file, new_file)
        
def rename_all(dataset_dir):
    topic_count = 0
    for domain in os.listdir(dataset_dir):
        #rename_files(dataset_dir+"/"+domain+"/doc_segs")
        for topic_dir in os.listdir(dataset_dir+"/"+domain+"/doc_links"):
            topic_count += 1
            rename_files(dataset_dir+"/"+domain+"/doc_links/"+topic_dir)
    print("Total topics: %d"%topic_count)
    
def count_topics_mod(dataset_dirs):
    topic_counts = {"html": 0, "ppt": 0, "pdf": 0, "video": 0}
    counted_topics = {"html": [], "ppt": [], "pdf": [], "video": []}
    for dataset_dir in dataset_dirs:
        docs_mod = {}
        for doc in os.listdir(dataset_dir+"/doc_segs"):
            doc_short_name = doc.split("_")
            doc_short_name = doc_short_name[0]+"_"+doc_short_name[1].replace(".txt", "")
            if "html" in doc:
                mod = "html"
            elif "ppt" in doc:
                mod = "ppt"
            elif "pdf" in doc:
                mod = "pdf"
            else:
                mod = "video"
            docs_mod[doc_short_name] = mod
        
        for topic in os.listdir(dataset_dir+"/doc_links"):
            for doc_seg in os.listdir(dataset_dir+"/doc_links/"+topic):
                doc_seg_short_name = doc_seg.split("_")
                doc_seg_short_name = doc_seg_short_name[0]+"_"+doc_seg_short_name[1].replace(".txt", "")
                if doc_seg_short_name in docs_mod:
                    mod = docs_mod[doc_seg_short_name]
                    if topic not in counted_topics[mod]:
                        topic_counts[mod]  += 1
                        counted_topics[mod].append(topic)
    print("Topic counts %s"%(str(topic_counts)))
        
#rename_all("/home/pjdrm/workspace/SegmentationScripts/src/mw_lecture")
count_topics_mod(["/home/pjdrm/workspace/TopicTrackingSegmentation/dataset/L02",
                  "/home/pjdrm/workspace/TopicTrackingSegmentation/dataset/L03",
                  "/home/pjdrm/workspace/TopicTrackingSegmentation/dataset/L06",
                  "/home/pjdrm/workspace/TopicTrackingSegmentation/dataset/L08",
                  "/home/pjdrm/workspace/TopicTrackingSegmentation/dataset/L10",
                  "/home/pjdrm/workspace/TopicTrackingSegmentation/dataset/L11",
                  "/home/pjdrm/workspace/TopicTrackingSegmentation/dataset/L20"])