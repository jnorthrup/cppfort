
/Users/jim/work/cppfort/micro-tests/results/memory/mem074-reference-parameter/mem074-reference-parameter_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z9incrementRi>:
100000360: b9400008    	ldr	w8, [x0]
100000364: 11000508    	add	w8, w8, #0x1
100000368: b9000008    	str	w8, [x0]
10000036c: d65f03c0    	ret

0000000100000370 <_main>:
100000370: 52800540    	mov	w0, #0x2a               ; =42
100000374: d65f03c0    	ret
