
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf013-ternary-assignment/cf013-ternary-assignment_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z23test_ternary_assignmenti>:
100000360: 531f7808    	lsl	w8, w0, #1
100000364: 4b0003e9    	neg	w9, w0
100000368: 53017d29    	lsr	w9, w9, #1
10000036c: 7100001f    	cmp	w0, #0x0
100000370: 5a89c500    	csneg	w0, w8, w9, gt
100000374: d65f03c0    	ret

0000000100000378 <_main>:
100000378: 52800280    	mov	w0, #0x14               ; =20
10000037c: d65f03c0    	ret
