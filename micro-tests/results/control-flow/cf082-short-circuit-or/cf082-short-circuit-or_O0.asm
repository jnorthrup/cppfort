
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf082-short-circuit-or/cf082-short-circuit-or_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15expensive_checkRi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: f90007e0    	str	x0, [sp, #0x8]
100000368: f94007e9    	ldr	x9, [sp, #0x8]
10000036c: b9400128    	ldr	w8, [x9]
100000370: 11000508    	add	w8, w8, #0x1
100000374: b9000128    	str	w8, [x9]
100000378: 52800008    	mov	w8, #0x0                ; =0
10000037c: 12000100    	and	w0, w8, #0x1
100000380: 910043ff    	add	sp, sp, #0x10
100000384: d65f03c0    	ret

0000000100000388 <__Z21test_short_circuit_orv>:
100000388: d10043ff    	sub	sp, sp, #0x10
10000038c: b9000fff    	str	wzr, [sp, #0xc]
100000390: 52800028    	mov	w8, #0x1                ; =1
100000394: 39002fe8    	strb	w8, [sp, #0xb]
100000398: b9400fe0    	ldr	w0, [sp, #0xc]
10000039c: 910043ff    	add	sp, sp, #0x10
1000003a0: d65f03c0    	ret

00000001000003a4 <_main>:
1000003a4: d10083ff    	sub	sp, sp, #0x20
1000003a8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003ac: 910043fd    	add	x29, sp, #0x10
1000003b0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003b4: 97fffff5    	bl	0x100000388 <__Z21test_short_circuit_orv>
1000003b8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003bc: 910083ff    	add	sp, sp, #0x20
1000003c0: d65f03c0    	ret
