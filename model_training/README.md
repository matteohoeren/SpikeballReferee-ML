# What is this repo about?

This is a python repository to train a machine learning model to predict if the ball hit the rim or net.
This (hopefully) solves an issue i came across playing spikeball with my friends. There would be some discussions
about wether it was a faulty hit or not so i decided to build a digital referee with the help of ✨ Machine Learning

## Collecting your own data

You can do so quite easily by utilizing the python program i've already built.
[Check this out this Repo](https://github.com/matteohoeren/spikeball-record-data)

If you don't want to go through the tedious process of gathering a significant amount of data, you can use mine or extend my data with yours.

## Data structure

You should have the following data structure in this python project:

```
main.py
rawdata/
   |--- net/
   |--- rim/
```

in each net and rim folder, you should have x-amount of csv files with the following columns

```
timestamp | x_accel | y_accel | z_accel
0,1742,1403,800
3,1450,1243,640
 ....
   .
   .
   .
```
