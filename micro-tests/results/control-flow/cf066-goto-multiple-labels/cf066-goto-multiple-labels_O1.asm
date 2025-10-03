
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf066-goto-multiple-labels/cf066-goto-multiple-labels_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_multiple_gotosi>:
100000360: 51000408    	sub	w8, w0, #0x1
100000364: 0b000809    	add	w9, w0, w0, lsl #2
100000368: 531f7929    	lsl	w9, w9, #1
10000036c: 71000d1f    	cmp	w8, #0x3
100000370: 1a9f3120    	csel	w0, w9, wzr, lo
100000374: d65f03c0    	ret

0000000100000378 <_main>:
100000378: 52800280    	mov	w0, #0x14               ; =20
10000037c: d65f03c0    	ret
