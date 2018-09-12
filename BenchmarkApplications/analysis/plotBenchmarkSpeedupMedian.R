library(data.table)
library(plyr)
library(reshape2)
library(ggplot2)
library(splitstackshape)

#bm <- "spmv"
#dataFile <- "bulk_data.txt"	

args <- commandArgs(trailingOnly = TRUE)

if (length(args)<2) {
  stop("Requires benchmark name as an argument! { mm, spmv, cfd }", call.=FALSE)
} else if (length(args)==2) {
  # default output file
  bm <- as.character(commandArgs(TRUE)[1]) 
  dataFile <- as.character(commandArgs(TRUE)[2]) 
}

saveDir <- "./plots"

violet <- c("darkviolet")
blue <- c("royalblue4")
green <- c("darkgreen")
turquoise <- c("turquoise4")

initTable<-read.csv(dataFile,header=FALSE,col.names=c('platform','benchmark','name','class','time'),sep=",",stringsAsFactors = FALSE)

# get median of times
dtavg <- aggregate(.~platform+benchmark+name+class,initTable,mean) 

# add column for speedup in a rather convoluted way
globalTable <- subset(dtavg, class %in% c("all global"))
dtavg <- subset(dtavg, !(class %in% c("unknown")))
drops <- c("class","name")
globalTable <- globalTable[,!(names(globalTable) %in% drops)]
colnames(globalTable)[which(names(globalTable) == "time")] <- "speedup"
dtavgNew <- merge(dtavg,globalTable,by.x=c("platform","benchmark"),by.y=c("platform","benchmark"),all.x=TRUE,all.y=TRUE)
dtavgNew$speedup <- dtavgNew$speedup / dtavgNew$time

# delete the unnecessary columns
drops <- c("name","time")
dtavgNew <- dtavgNew[ , !(names(dtavgNew) %in% drops)]

# get only the benchmark "of interest"
benchmarkTable <- subset(dtavgNew, benchmark %in% c(bm))
benchmarkTable <- subset(benchmarkTable, !(class %in% c("all global")))

#benchmarkTable <- subset(benchmarkTable, !(platform %in% c("Maxwell")))

melted <- melt(benchmarkTable)
nameSave <- paste(saveDir,paste(bm,paste("_speedup.pdf",sep=""),sep=""),sep="/")

meltedPlot <- ggplot(na.omit(melted[,]), aes(x=class, y=value, fill=variable)) + 
	theme_bw() +
	geom_bar(position=position_dodge(),stat='identity') +
	geom_bar(position=position_dodge(),stat='identity',color="Black",size=.1,show_guide=FALSE) +	
	theme(plot.title = element_text(hjust=0.5),
   	      strip.text.x = element_text(size = 18),
	      axis.title.x = element_text(face="bold",size=18),#, margin=margin(t = 10, r = 0, b = 0, l = 0)),
                   axis.title.y = element_text(face="bold",size=18),
                   axis.text.x=element_text(face="bold",angle=45,hjus=1,vjus=1,size=18, margin=margin(t = 0, r = 0, b = 20, l = 0)),
                   axis.text.y=element_text(face="bold",size=18),
                   axis.ticks.y = element_blank(),
                   strip.text=element_text(size=7),
                   strip.background = element_rect(
                       size = 0.2),
                   panel.margin = unit(0.1, "lines"),
                   #legend.margin = unit(c(0,0,-1.6,0), "mm"),#legend.margin = unit(c(1,-48,0,0), "mm"),
                   #legend.key.size = unit(9,"pt"),
                   #legend.title = element_text(size=8),
                  # legend.text=element_text(size=8),
                  # legend.position="top",
                   plot.margin = unit(c(3,3,3,3), "mm")) +	
		   theme(legend.position=c(-1,-1)) +
		   xlab("Memory Type [ Platform ]") + 
		   ylab("Speedup") +
		   scale_fill_brewer(palette = "Blues") +
		   facet_wrap(~platform,nrow=1) +
#		   coord_cartesian(ylim = c(0.7, 1.3))  + # for CFD only !!
		   geom_hline(yintercept=1,linetype="dashed",color="red")


ggsave(nameSave, plot = meltedPlot,width = 185, height = 95, units="mm", family= "Times",pointsize=20, useDingbats=FALSE)


