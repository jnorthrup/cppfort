
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf054-switch-enum/cf054-switch-enum_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_switch_enum5Color>:
100000360: 52800148    	mov	w8, #0xa                ; =10
100000364: 52800149    	mov	w9, #0xa                ; =10
100000368: 1b082408    	madd	w8, w0, w8, w9
10000036c: 71000c1f    	cmp	w0, #0x3
100000370: 1a9f3100    	csel	w0, w8, wzr, lo
100000374: d65f03c0    	ret

0000000100000378 <_main>:
100000378: 52800280    	mov	w0, #0x14               ; =20
10000037c: d65f03c0    	ret
