library(ggplot2)
library(immunarch)
library(dplyr)
library(stringr)
library(forcats)
library(reshape2)
library(purrr)
library(plotly)

plot_unique <- function(data_subset, title_text, show_x_text = FALSE) {

  uniq <- repExplore(data_subset, .method = "volume")
  uniq_df <- data.frame(uniq)

  p <- ggplot(uniq_df, aes(x = Sample, y = Volume)) +
    geom_bar(stat = "identity") +
    theme_bw() +
    labs(title = title_text,
         x = "Sample",
         y = "Count") +
    scale_y_continuous(labels = scales::comma)
  axis_theme <- if (show_x_text) {
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5))
  } else {
    theme(axis.text.x = element_blank())
  }

  p <- p + axis_theme + theme(panel.grid = element_blank())

  list(plot = p, values = uniq_df$Volume)
}

plot_top <- function(data_subset, title_text, show_x_text = FALSE) {
  top <- repClonality(data_subset, .method = "top", .head= c(10,100,1000,3000,10000,30000)) 

top_df <- data.frame(top)
top_n <- mutate(top_df, "[11:100)" = X100 - X10, 
                  "[101:1000)" = X1000 - X100, 
                  "[1001:3000)" = X3000 - X1000, 
                  "[3001:10000)" = X10000 - X3000,
                  "[10001:30000)" = X30000 - X10000,
                  "Patient_ID" = rownames(top_df),
                  Num = as.numeric(map(str_split(Patient_ID, regex("_"),2),1)),
                  Patient_ID = fct_reorder(Patient_ID, Num))

top_n1 <- top_n[,c("Patient_ID","X10","[11:100)", "[101:1000)","[1001:3000)","[3001:10000)","[10001:30000)")] 
top_n1 <- top_n1 %>% rename( "[1:10)" = "X10")

test <- reshape2::melt(top_n1)
p <- ggplot(test, aes(x = Patient_ID, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "stack")+
  theme_bw() +
  labs(title = title_text, x = "Sample", y = "Occupied repertoire space") + 
  guides(fill=guide_legend(title="Clonotype indices")) +
  scale_fill_brewer(palette = "RdYlBu")

  axis_theme <- if (show_x_text) {
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5))
  } else {
    theme(axis.text.x = element_blank())
  }

  p <- p + axis_theme + theme(panel.grid = element_blank())

  list(plot = p, values = top_n1[["[1:10)"]])
}

plot_rare <- function(data_subset, title_text, show_x_text = FALSE) {
  rare <- repClonality(data_subset, .method = "rare")

rare_df <- data.frame(rare)
rare_n <- mutate(rare_df, "2 - 3" = X3 - X1, 
                  "4 - 10" = X10 - X3, 
                  "11 - 30" = X30 - X10, 
                  "31 - 100" = X100 - X30,
                  "101 - MAX" = MAX - X100,
                  "Patient_ID" = rownames(rare_df),
                  Num = as.numeric(map(str_split(Patient_ID, regex("_"),2),1)),
                  Patient_ID = fct_reorder(Patient_ID, Num))
rare_n1 <- rare_n[,c("Patient_ID","X1","2 - 3", "4 - 10", "11 - 30","31 - 100","101 - MAX")]
rare_n1 = rare_n1 %>% rename( "1" = "X1")
test_rare <- reshape2::melt(rare_n1)

p <- ggplot(test_rare,aes(x=Patient_ID,y=value,fill=variable))+
  geom_bar(stat='identity',position='stack')+
  theme_bw()+
  labs(title = title_text, x = "Sample", y = "Occupied repertoire space") + 
  guides(fill=guide_legend(title="Clonotype counts"))+
  scale_fill_brewer(palette = "RdYlBu")

axis_theme <- if (show_x_text) {
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5), complete = F)
  } else {
    theme(axis.text.x = element_blank())
  }

  p <- p + axis_theme + theme(panel.grid = element_blank())

  list(plot = p, values = rare_n1[["1"]])
}

plot_diversity <- function(data_subset, method, title_text, y_label, show_x_text = FALSE) {

  div <- repDiversity(data_subset, .method = method, .verbose = FALSE)
  # if method == "gini", we want to turn object to df, 
  # and rename columns to "Sample" and "Value"
  if (method == "gini") {
    div <- data.frame(div)
    div$Sample <- rownames(div)
    div$Value <- div$gini_h
    div$gini_h <- NULL
  } 
  div$Num <- as.numeric(stringr::str_extract(div$Sample, "^[0-9]+"))
  div$Patient_ID <- forcats::fct_reorder(div$Sample, div$Num)

  p <- ggplot(div, aes(Patient_ID, Value)) +
    geom_col() +
    theme_bw() +
    labs(title = title_text,
         x = "Sample",
         y = y_label) +
    theme(legend.position = "none")

  axis_theme <- if (show_x_text) {
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5))
    } else {
      theme(axis.text.x = element_blank())
    }

  p <- p + axis_theme + theme(panel.grid = element_blank())

  list(plot = p, values = div$Value)
}

plot_overlap <- function(sub) {

  public <- repOverlap(sub, .method = "public", .verbose = FALSE, .col = "aa")
  public_df <- as.data.frame(public)

  ordered_names <- names(public_df)[order(gsub(".*_", "", names(public_df)))]

  public_df <- public_df[ordered_names, ordered_names]

  plot_ly(
    x = colnames(public_df),
    y = rownames(public_df),
    z = as.matrix(public_df),
    type = "heatmap",
    zauto = FALSE,
    zmin = 0,
    zmax = 800
  ) |>
    layout(title = "Repertoire Overlap Blood OC and HD", xaxis = list(title = 'Sample', tickangle=-90), yaxis = list(title = 'Sample'))
}
