
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf081-short-circuit-and/cf081-short-circuit-and_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15expensive_checkRi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: f90007e0    	str	x0, [sp, #0x8]
100000368: f94007ea    	ldr	x10, [sp, #0x8]
10000036c: b9400149    	ldr	w9, [x10]
100000370: 52800028    	mov	w8, #0x1                ; =1
100000374: 11000529    	add	w9, w9, #0x1
100000378: b9000149    	str	w9, [x10]
10000037c: 12000100    	and	w0, w8, #0x1
100000380: 910043ff    	add	sp, sp, #0x10
100000384: d65f03c0    	ret

0000000100000388 <__Z22test_short_circuit_andv>:
100000388: d10043ff    	sub	sp, sp, #0x10
10000038c: b9000fff    	str	wzr, [sp, #0xc]
100000390: 39002fff    	strb	wzr, [sp, #0xb]
100000394: b9400fe0    	ldr	w0, [sp, #0xc]
100000398: 910043ff    	add	sp, sp, #0x10
10000039c: d65f03c0    	ret

00000001000003a0 <_main>:
1000003a0: d10083ff    	sub	sp, sp, #0x20
1000003a4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003a8: 910043fd    	add	x29, sp, #0x10
1000003ac: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003b0: 97fffff6    	bl	0x100000388 <__Z22test_short_circuit_andv>
1000003b4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003b8: 910083ff    	add	sp, sp, #0x20
1000003bc: d65f03c0    	ret
