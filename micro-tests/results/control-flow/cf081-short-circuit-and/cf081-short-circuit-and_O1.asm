
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf081-short-circuit-and/cf081-short-circuit-and_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15expensive_checkRi>:
100000360: b9400008    	ldr	w8, [x0]
100000364: 11000508    	add	w8, w8, #0x1
100000368: b9000008    	str	w8, [x0]
10000036c: 52800020    	mov	w0, #0x1                ; =1
100000370: d65f03c0    	ret

0000000100000374 <__Z22test_short_circuit_andv>:
100000374: 52800000    	mov	w0, #0x0                ; =0
100000378: d65f03c0    	ret

000000010000037c <_main>:
10000037c: 52800000    	mov	w0, #0x0                ; =0
100000380: d65f03c0    	ret
