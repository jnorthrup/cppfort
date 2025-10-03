
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf089-short-circuit-function-calls/cf089-short-circuit-function-calls_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z5func1v>:
100000360: 52800000    	mov	w0, #0x0                ; =0
100000364: d65f03c0    	ret

0000000100000368 <__Z5func2v>:
100000368: 52800020    	mov	w0, #0x1                ; =1
10000036c: d65f03c0    	ret

0000000100000370 <__Z5func3v>:
100000370: 52800040    	mov	w0, #0x2                ; =2
100000374: d65f03c0    	ret

0000000100000378 <__Z27test_function_short_circuitv>:
100000378: d10083ff    	sub	sp, sp, #0x20
10000037c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000380: 910043fd    	add	x29, sp, #0x10
100000384: 97fffff7    	bl	0x100000360 <__Z5func1v>
100000388: 52800028    	mov	w8, #0x1                ; =1
10000038c: b81fc3a8    	stur	w8, [x29, #-0x4]
100000390: 35000180    	cbnz	w0, 0x1000003c0 <__Z27test_function_short_circuitv+0x48>
100000394: 14000001    	b	0x100000398 <__Z27test_function_short_circuitv+0x20>
100000398: 97fffff4    	bl	0x100000368 <__Z5func2v>
10000039c: 52800028    	mov	w8, #0x1                ; =1
1000003a0: b81fc3a8    	stur	w8, [x29, #-0x4]
1000003a4: 350000e0    	cbnz	w0, 0x1000003c0 <__Z27test_function_short_circuitv+0x48>
1000003a8: 14000001    	b	0x1000003ac <__Z27test_function_short_circuitv+0x34>
1000003ac: 97fffff1    	bl	0x100000370 <__Z5func3v>
1000003b0: 71000008    	subs	w8, w0, #0x0
1000003b4: 1a9f07e8    	cset	w8, ne
1000003b8: b81fc3a8    	stur	w8, [x29, #-0x4]
1000003bc: 14000001    	b	0x1000003c0 <__Z27test_function_short_circuitv+0x48>
1000003c0: b85fc3a8    	ldur	w8, [x29, #-0x4]
1000003c4: 12000100    	and	w0, w8, #0x1
1000003c8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003cc: 910083ff    	add	sp, sp, #0x20
1000003d0: d65f03c0    	ret

00000001000003d4 <_main>:
1000003d4: d10083ff    	sub	sp, sp, #0x20
1000003d8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003dc: 910043fd    	add	x29, sp, #0x10
1000003e0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003e4: 97ffffe5    	bl	0x100000378 <__Z27test_function_short_circuitv>
1000003e8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003ec: 910083ff    	add	sp, sp, #0x20
1000003f0: d65f03c0    	ret
