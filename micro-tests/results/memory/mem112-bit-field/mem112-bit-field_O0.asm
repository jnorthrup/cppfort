
/Users/jim/work/cppfort/micro-tests/results/memory/mem112-bit-field/mem112-bit-field_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z14test_bit_fieldv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: 79401be8    	ldrh	w8, [sp, #0xc]
100000368: 121d7108    	and	w8, w8, #0xfffffff8
10000036c: 528000e9    	mov	w9, #0x7                ; =7
100000370: 32000908    	orr	w8, w8, #0x7
100000374: 79001be8    	strh	w8, [sp, #0xc]
100000378: 79401be8    	ldrh	w8, [sp, #0xc]
10000037c: 12186908    	and	w8, w8, #0xffffff07
100000380: 321d1108    	orr	w8, w8, #0xf8
100000384: 79001be8    	strh	w8, [sp, #0xc]
100000388: 394033e8    	ldrb	w8, [sp, #0xc]
10000038c: 32185d08    	orr	w8, w8, #0xffffff00
100000390: 79001be8    	strh	w8, [sp, #0xc]
100000394: 79401be8    	ldrh	w8, [sp, #0xc]
100000398: 0a090108    	and	w8, w8, w9
10000039c: 79401be9    	ldrh	w9, [sp, #0xc]
1000003a0: 53037d29    	lsr	w9, w9, #3
1000003a4: 12001129    	and	w9, w9, #0x1f
1000003a8: 0b090100    	add	w0, w8, w9
1000003ac: 910043ff    	add	sp, sp, #0x10
1000003b0: d65f03c0    	ret

00000001000003b4 <_main>:
1000003b4: d10083ff    	sub	sp, sp, #0x20
1000003b8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003bc: 910043fd    	add	x29, sp, #0x10
1000003c0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003c4: 97ffffe7    	bl	0x100000360 <__Z14test_bit_fieldv>
1000003c8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003cc: 910083ff    	add	sp, sp, #0x20
1000003d0: d65f03c0    	ret
