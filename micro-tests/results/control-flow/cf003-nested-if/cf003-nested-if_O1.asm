
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf003-nested-if/cf003-nested-if_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z14test_nested_ifii>:
100000360: 7100043f    	cmp	w1, #0x1
100000364: 52800028    	mov	w8, #0x1                ; =1
100000368: 1a88a508    	cinc	w8, w8, lt
10000036c: 52800069    	mov	w9, #0x3                ; =3
100000370: 7100001f    	cmp	w0, #0x0
100000374: 1a89c100    	csel	w0, w8, w9, gt
100000378: d65f03c0    	ret

000000010000037c <_main>:
10000037c: 52800020    	mov	w0, #0x1                ; =1
100000380: d65f03c0    	ret
