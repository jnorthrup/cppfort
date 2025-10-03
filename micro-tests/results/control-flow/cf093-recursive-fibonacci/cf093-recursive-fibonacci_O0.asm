
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf093-recursive-fibonacci/cf093-recursive-fibonacci_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z9fibonaccii>:
100000360: d10083ff    	sub	sp, sp, #0x20
100000364: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000368: 910043fd    	add	x29, sp, #0x10
10000036c: b9000be0    	str	w0, [sp, #0x8]
100000370: b9400be8    	ldr	w8, [sp, #0x8]
100000374: 71000508    	subs	w8, w8, #0x1
100000378: 540000ac    	b.gt	0x10000038c <__Z9fibonaccii+0x2c>
10000037c: 14000001    	b	0x100000380 <__Z9fibonaccii+0x20>
100000380: b9400be8    	ldr	w8, [sp, #0x8]
100000384: b81fc3a8    	stur	w8, [x29, #-0x4]
100000388: 1400000d    	b	0x1000003bc <__Z9fibonaccii+0x5c>
10000038c: b9400be8    	ldr	w8, [sp, #0x8]
100000390: 71000500    	subs	w0, w8, #0x1
100000394: 97fffff3    	bl	0x100000360 <__Z9fibonaccii>
100000398: b90007e0    	str	w0, [sp, #0x4]
10000039c: b9400be8    	ldr	w8, [sp, #0x8]
1000003a0: 71000900    	subs	w0, w8, #0x2
1000003a4: 97ffffef    	bl	0x100000360 <__Z9fibonaccii>
1000003a8: aa0003e8    	mov	x8, x0
1000003ac: b94007e0    	ldr	w0, [sp, #0x4]
1000003b0: 0b080008    	add	w8, w0, w8
1000003b4: b81fc3a8    	stur	w8, [x29, #-0x4]
1000003b8: 14000001    	b	0x1000003bc <__Z9fibonaccii+0x5c>
1000003bc: b85fc3a0    	ldur	w0, [x29, #-0x4]
1000003c0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003c4: 910083ff    	add	sp, sp, #0x20
1000003c8: d65f03c0    	ret

00000001000003cc <_main>:
1000003cc: d10083ff    	sub	sp, sp, #0x20
1000003d0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003d4: 910043fd    	add	x29, sp, #0x10
1000003d8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003dc: 52800140    	mov	w0, #0xa                ; =10
1000003e0: 97ffffe0    	bl	0x100000360 <__Z9fibonaccii>
1000003e4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003e8: 910083ff    	add	sp, sp, #0x20
1000003ec: d65f03c0    	ret
