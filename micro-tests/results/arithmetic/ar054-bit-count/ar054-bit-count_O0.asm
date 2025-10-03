
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar054-bit-count/ar054-bit-count_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z14test_bit_countj>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fe0    	str	w0, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 14000001    	b	0x100000370 <__Z14test_bit_countj+0x10>
100000370: b9400fe8    	ldr	w8, [sp, #0xc]
100000374: 34000168    	cbz	w8, 0x1000003a0 <__Z14test_bit_countj+0x40>
100000378: 14000001    	b	0x10000037c <__Z14test_bit_countj+0x1c>
10000037c: b9400fe8    	ldr	w8, [sp, #0xc]
100000380: 12000109    	and	w9, w8, #0x1
100000384: b9400be8    	ldr	w8, [sp, #0x8]
100000388: 0b090108    	add	w8, w8, w9
10000038c: b9000be8    	str	w8, [sp, #0x8]
100000390: b9400fe8    	ldr	w8, [sp, #0xc]
100000394: 53017d08    	lsr	w8, w8, #1
100000398: b9000fe8    	str	w8, [sp, #0xc]
10000039c: 17fffff5    	b	0x100000370 <__Z14test_bit_countj+0x10>
1000003a0: b9400be0    	ldr	w0, [sp, #0x8]
1000003a4: 910043ff    	add	sp, sp, #0x10
1000003a8: d65f03c0    	ret

00000001000003ac <_main>:
1000003ac: d10083ff    	sub	sp, sp, #0x20
1000003b0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003b4: 910043fd    	add	x29, sp, #0x10
1000003b8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003bc: 52801540    	mov	w0, #0xaa               ; =170
1000003c0: 97ffffe8    	bl	0x100000360 <__Z14test_bit_countj>
1000003c4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003c8: 910083ff    	add	sp, sp, #0x20
1000003cc: d65f03c0    	ret
