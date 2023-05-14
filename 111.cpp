#include <iostream>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
using namespace std;

int question_number=1;//题目数量,默认为1
int symbol_number=1;//运算符种类数,默认为1
char symbol_type[5]={0};//运算符种类，默认为空
int max_number=1;//最大数，默认为1
int brackets=0;//是否有括号（1表示有括号，0表示没有括号）,默认为0
int decimal=0;//是否有小数（1表示有小数，0表示没有小数）,默认为0
int print_style=0;//用户选择输出方式（1表示输出到文件，0表示输出屏幕）,默认为0
float number[5]={0};//使用的数字,默认为0
char symbol[4]={0};//使用的运算符，默认为空

void menu(void)//主菜单界面
{
    printf("**********************************************************************************\n");
    printf("\n");
    printf("                              四则运算习题自动生成器                              \n");
    printf("\n");
    printf("                              请按提示设置练习题类型                              \n");
    printf("\n");
    printf("**********************************************************************************\n");
}

void parameter(void)//参数设置
{
    printf("请设置题目数量（正整数）\n");
    scanf("%d",&question_number);
    if(question_number<=0)
    {
        printf("请重新设置题目数量（正整数）\n");
        scanf("%d",&question_number);
    }
    printf("请设置运算符种类数(1~4之间的整数)\n");
    scanf("%d",&symbol_number);
    if((symbol_number<1)||(symbol_number>4))
    {
        printf("请重新设置运算符种类数(1~4之间的整数)\n");
        scanf("%d",&symbol_number);
    }
    printf("请连续输入使用的运算符（在“+”，“-”，“*”，“/”中选择输入）\n");
    scanf("%s",symbol_type);
    printf("请设置最大数（正整数）\n");
    scanf("%d",&max_number);
    if(max_number<=0)
    {
        printf("请重新设置最大数（正整数）\n");
        scanf("%d",&max_number);
    }
    printf("请选择是否有括号（0表示没有括号，其他数字表示有括号）\n");
    scanf("%d",&brackets);
    printf("请选择是否有小数（0表示没有小数，其他数字表示有小数）\n");
    scanf("%d",&decimal);
    printf("请选择输出方式（0表示输出到屏幕，其他数字表示输出到文件）\n");
    scanf("%d",&print_style);
}

void randomnumber(void)//随机产生数字
{
	for (int i=0;i<4;i++)
	{
	    if(decimal==0)//随机产生整数
        {
            number[i]=rand()%(max_number+1);
        }
        else//随机产生小数
        {
            number[i]=rand()%max_number;
            number[i]=number[i]+rand()%100*0.01;
        }
	}
}

void randomsymbol(void)//随机产生符号
{
	for (int i=0;i<3;i++)
	{
        symbol[i]=symbol_type[rand()%symbol_number];
	}
}

void print_expression(void)//输出函数
{
    int brackets_place;//括号位置
    FILE *fp;//文件指针
    fp=fopen("Four operational exercises.txt","w");//打开文件写操作
    for(int m=0;m<question_number;m++)
    {
        randomnumber();//随机产生数字
        randomsymbol();//随机产生符号
        for(int i=0;i<3;i++)
        {
            if((symbol[i]=='/')&&(number[i+1]==0))//除数为0时产生错误，为避免这种错误把除数加一
            {
                number[i+1]=number[i+1]+1;
            }
        }
        if(decimal==0)//不是小数
        {
            if(print_style==0)//输出到屏幕
            {
                if(brackets==0)//没有括号
                {
                    printf("%.0f%c%.0f%c%.0f%c%.0f=\n",number[0],symbol[0],number[1],symbol[1],number[2],symbol[2],number[3]);
                }
                else//有括号
                {
                    brackets_place=rand()%3;
                    if(brackets_place==0)//括号在第一个数和第二个数上
                    {
                        printf("（%.0f%c%.0f）%c%.0f%c%.0f=\n",number[0],symbol[0],number[1],symbol[1],number[2],symbol[2],number[3]);
                    }
                    if(brackets_place==1)//括号在第二个数和第三个数上
                    {
                        printf("%.0f%c(%.0f%c%.0f)%c%.0f=\n",number[0],symbol[0],number[1],symbol[1],number[2],symbol[2],number[3]);
                    }
                    if(brackets_place==2)//括号在第三个数和第四个数上
                    {
                        printf("%.0f%c%.0f%c(%.0f%c%.0f)=\n",number[0],symbol[0],number[1],symbol[1],number[2],symbol[2],number[3]);
                    }
                }
            }
            else//输出到文件
            {
                if(brackets==0)//没有括号
                {
                    fprintf(fp,"%.0f%c%.0f%c%.0f%c%.0f=\n",number[0],symbol[0],number[1],symbol[1],number[2],symbol[2],number[3]);
                }
                else//有括号
                {
                    brackets_place=rand()%3;
                    if(brackets_place==0)//括号在第一个数和第二个数上
                    {
                        fprintf(fp,"（%.0f%c%.0f）%c%.0f%c%.0f=\n",number[0],symbol[0],number[1],symbol[1],number[2],symbol[2],number[3]);
                    }
                    if(brackets_place==1)//括号在第二个数和第三个数上
                    {
                        fprintf(fp,"%.0f%c(%.0f%c%.0f)%c%.0f=\n",number[0],symbol[0],number[1],symbol[1],number[2],symbol[2],number[3]);
                    }
                    if(brackets_place==2)//括号在第三个数和第四个数上
                    {
                        fprintf(fp,"%.0f%c%.0f%c(%.0f%c%.0f)=\n",number[0],symbol[0],number[1],symbol[1],number[2],symbol[2],number[3]);
                    }
                }
            }
        }
        else//是小数
        {
            if(print_style==0)//输出到屏幕
            {
                if(brackets==0)//没有括号
                {
                    printf("%.2f%c%.2f%c%.2f%c%.2f=\n",number[0],symbol[0],number[1],symbol[1],number[2],symbol[2],number[3]);
                }
                else//有括号
                {
                    brackets_place=rand()%3;
                    if(brackets_place==0)//括号在第一个数和第二个数上
                    {
                        printf("（%.2f%c%.2f）%c%.2f%c%.2f=\n",number[0],symbol[0],number[1],symbol[1],number[2],symbol[2],number[3]);
                    }
                    if(brackets_place==1)//括号在第二个数和第三个数上
                    {
                        printf("%.2f%c(%.2f%c%.2f)%c%.2f=\n",number[0],symbol[0],number[1],symbol[1],number[2],symbol[2],number[3]);
                    }
                    if(brackets_place==2)//括号在第三个数和第四个数上
                    {
                        printf("%.2f%c%.2f%c(%.2f%c%.2f)=\n",number[0],symbol[0],number[1],symbol[1],number[2],symbol[2],number[3]);
                    }
                }
            }
            else//输出到文件
            {
                if(brackets==0)//没有括号
                {
                    fprintf(fp,"%.2f%c%.2f%c%.2f%c%.2f=\n",number[0],symbol[0],number[1],symbol[1],number[2],symbol[2],number[3]);
                }
                else//有括号
                {
                    brackets_place=rand()%3;
                    if(brackets_place==0)//括号在第一个数和第二个数上
                    {
                        fprintf(fp,"（%.2f%c%.2f）%c%.2f%c%.2f=\n",number[0],symbol[0],number[1],symbol[1],number[2],symbol[2],number[3]);
                    }
                    if(brackets_place==1)//括号在第二个数和第三个数上
                    {
                        fprintf(fp,"%.2f%c(%.2f%c%.2f)%c%.2f=\n",number[0],symbol[0],number[1],symbol[1],number[2],symbol[2],number[3]);
                    }
                    if(brackets_place==2)//括号在第三个数和第四个数上
                    {
                        fprintf(fp,"%.2f%c%.2f%c(%.2f%c%.2f)=\n",number[0],symbol[0],number[1],symbol[1],number[2],symbol[2],number[3]);
                    }
                }
            }
        }
    }
    fclose(fp);//关闭文件
}

int main()
{
    menu();//主菜单界面
    parameter();//参数设置
    srand((int)time(0));//保证每次产生的随机数不同
    print_expression();//输出函数
    return 0;
}
