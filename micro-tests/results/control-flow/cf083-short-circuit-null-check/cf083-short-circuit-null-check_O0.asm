
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf083-short-circuit-null-check/cf083-short-circuit-null-check_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_null_checkPi>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: f90003e0    	str	x0, [sp]
100000368: f94003e8    	ldr	x8, [sp]
10000036c: b4000168    	cbz	x8, 0x100000398 <__Z15test_null_checkPi+0x38>
100000370: 14000001    	b	0x100000374 <__Z15test_null_checkPi+0x14>
100000374: f94003e8    	ldr	x8, [sp]
100000378: b9400108    	ldr	w8, [x8]
10000037c: 71000108    	subs	w8, w8, #0x0
100000380: 540000cd    	b.le	0x100000398 <__Z15test_null_checkPi+0x38>
100000384: 14000001    	b	0x100000388 <__Z15test_null_checkPi+0x28>
100000388: f94003e8    	ldr	x8, [sp]
10000038c: b9400108    	ldr	w8, [x8]
100000390: b9000fe8    	str	w8, [sp, #0xc]
100000394: 14000003    	b	0x1000003a0 <__Z15test_null_checkPi+0x40>
100000398: b9000fff    	str	wzr, [sp, #0xc]
10000039c: 14000001    	b	0x1000003a0 <__Z15test_null_checkPi+0x40>
1000003a0: b9400fe0    	ldr	w0, [sp, #0xc]
1000003a4: 910043ff    	add	sp, sp, #0x10
1000003a8: d65f03c0    	ret

00000001000003ac <_main>:
1000003ac: d10083ff    	sub	sp, sp, #0x20
1000003b0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003b4: 910043fd    	add	x29, sp, #0x10
1000003b8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003bc: 910023e0    	add	x0, sp, #0x8
1000003c0: 52800548    	mov	w8, #0x2a               ; =42
1000003c4: b9000be8    	str	w8, [sp, #0x8]
1000003c8: 97ffffe6    	bl	0x100000360 <__Z15test_null_checkPi>
1000003cc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003d0: 910083ff    	add	sp, sp, #0x20
1000003d4: d65f03c0    	ret
