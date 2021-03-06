#!/usr/bin/env Rscript


# Instal these packages for R
# install.packages("plyr")
# install.packages("ggplot2")
# install.packages("grid")
# install.packages("RColorBrewer")
# install.packages("scales")
# install.packages("sqldf")
# install.packages("optparse")


library(plyr)

library(ggplot2)
library(grid)
library(RColorBrewer)
library(scales)
library(sqldf)
library("optparse")

# Some of the parameters used for graph
GPUList <- list("Volta", "Pascal", "Maxwell", "Kepler")
GPUBlockSizeList <- list("4", "8","16","32")
DataSizeList <- list('1024', '2048', '4096')
DataSizeName <- list('Small', 'Medium', 'Large')

ExecutableList <- list('./heat_2d.out')

# Parser function for parsing the arguments
option_list = list(
  make_option(c("-d", "--dir"), type="character", default=NULL, 
              help="Root directory for the GPU data", metavar="character"),
  make_option(c("-o", "--out"), type="character", default="out.txt", 
              help="output file name [default= %default]", metavar="character")
); 
 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);


# set the working directory to the value passed to option -d (--dir)
if (is.null(opt$dir)){
  print_help(opt_parser)
  stop("At least one argument must be supplied (directory).n", call.=FALSE)
}
setwd(opt$dir)
data = data.frame()

# Open files for differnt GPU's and combine them in one data frame
for (gpu in GPUList)
{
  # Set the appropiate file name
  filePath <- paste(gpu, "/Normalized_Exec_Data-", gpu,".csv", sep="")
  # print(filePath)
  tmpData <- read.csv(filePath, sep=",", header=T)
  tmpData$GPU <- gpu
  data = rbind(data, tmpData)
  print(data)
}


data$GPUBlockSize <- as.character(data$GPUBlockSize)
# Keep only data with the Block size that are part of GPUBlockSizeList
# The last ","  is necessary
data <- data[data$GPUBlockSize %in% GPUBlockSizeList,]


data$GPUBlockSize <- factor(data$GPUBlockSize,levels=GPUBlockSizeList)




# ################# @AppSpecific ######################## #

data$DataSize <- as.factor(as.character(data$DataSize))

# Keep data related to needed executables
data <- data[data$Executable %in% ExecutableList,]

# For 3 different data types
# We'll create 3 different plots


subsetData = list()
idx=1
for (dataSet in DataSizeList) 
{
  subsetData[[idx]] <- subset(data, (DataSize==dataSet))
  idx <- idx + 1
}

# @TODO, for new apps: Choose the appropiate memory type
plot.change.labels <- function(data) {
    levels(data$Executable)[levels(data$Executable)=="./heat.out"]         <- "Texture Memory"
    levels(data$Executable)[levels(data$Executable)=="./heat_globalmem.out"]         <- "Global Memory"
    levels(data$Executable)[levels(data$Executable)=="./heat_2d.out"]         <- "Texture Memory"
    levels(data$Executable)[levels(data$Executable)=="./heat_2d_globalmem.out"]         <- "Global Memory"
    return(data)
}

##########################################################

plot.memory.impact <- function(data, idx) {
    data <- plot.change.labels(data)
    theme_set( theme_bw() )
    p <- ggplot(data,
                aes(x=GPUBlockSize,
                    y=SpeedUp,
                    fill=Executable,
                    type=Executable))
    p <- p + geom_bar(position=position_dodge(),
                      stat='identity')
    p <- p+ geom_bar(position=position_dodge(),
                     stat='identity',
                     color="Black",
                     size=.1,
                     show_guide=FALSE)
    p <- p + ggtitle(paste("Data Size - ", idx))
    p <- p + ylab("Speedup")
    p <- p + xlab("GPU Thread Block Size")
    p <- p + theme(plot.title = element_text(hjust=0.5), 
                   axis.title.x = element_text(face="bold",size=10, margin=margin(t = 10, r = 0, b = 0, l = 0)),
                   axis.title.y = element_text(face="bold",size=10),
                   axis.text.x=element_text(face="bold",angle=45,hjus=1,vjus=1,size=10, margin=margin(t = 0, r = 0, b = 20, l = 0)),
                   axis.text.y=element_text(face="bold",size=10),
                   axis.ticks.y = element_blank(),
                   strip.text=element_text(size=7),
                   strip.background = element_rect(
                       size = 0.2),
                   panel.margin = unit(0.1, "lines"),
                   legend.margin = unit(c(0,0,-1.6,0), "mm"),#legend.margin = unit(c(1,-48,0,0), "mm"),
                   legend.key.size = unit(9,"pt"),
                   legend.title = element_text(size=8),
                   legend.text=element_text(size=8),
                   legend.position="top",
                   plot.margin = unit(c(1.5,1,1,0.5), "mm"))#plot.margin = unit(c(-5.5,1,-1,0), "mm"))
    # p <- p + scale_y_continuous(labels=comma)
    p <- p + scale_y_continuous(limits=c(0, 2))
    DefaultSpeedUp=1
    p <- p + geom_hline(yintercept=DefaultSpeedUp, linetype="dashed", color = "red")
    #p <- p + scale_x_continuous(labels=comma)
    # p <- p + scale_x_discrete(labels=c("8",
                                  # expression(paste("16")),
                                  # expression(paste("24")),
                                  # expression(paste("32"))))
    
    p <- p + scale_fill_brewer(name="Memory Type:",
                               palette = "Reds")
    p <- p + facet_wrap(~GPU,nrow=1)
    g <- ggplotGrob(p)
    g$heights[[7]] = unit(8,"pt")
    g$heights[[11]] = unit(8,"pt")
    grid.draw(g)   
    #return(p)
}

# ################# @AppSpecific ######################## #

# @TODO, for new apps: Choose the appropiate ouput file name and data size

filename="Texture_Mem_2D_Impact_Across_GPUBlock_Across_GPUs_"
i=1
for (ds in DataSizeList)
{
    fname <- paste(filename, ds, '.pdf', sep = "")
    pdf(fname,
    height=2.6,
    width=3.2,
    family="Times",
    pointsize=20,
    useDingbats=FALSE)
    plot.memory.impact(subsetData[[i]], DataSizeName[[i]])
    print(DataSizeName[[i]])
    print(DataSizeList[[i]])
    dev.off()
    # embedFonts(filename,
           # options="-dSubsetFonts=true -dEmbedAllFonts=true -dPDFSETTINGS=/printer -dUseCIEColor")
    i <- i+1

}


