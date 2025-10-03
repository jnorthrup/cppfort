
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf013-ternary-assignment/cf013-ternary-assignment_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z23test_ternary_assignmenti>:
100000360: 0b407c08    	add	w8, w0, w0, lsr #31
100000364: 13017d08    	asr	w8, w8, #1
100000368: 531f7809    	lsl	w9, w0, #1
10000036c: 7100041f    	cmp	w0, #0x1
100000370: 1a89b100    	csel	w0, w8, w9, lt
100000374: d65f03c0    	ret

0000000100000378 <_main>:
100000378: 52800280    	mov	w0, #0x14               ; =20
10000037c: d65f03c0    	ret
