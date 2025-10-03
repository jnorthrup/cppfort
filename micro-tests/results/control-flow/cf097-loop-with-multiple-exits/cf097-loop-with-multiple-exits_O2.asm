
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf097-loop-with-multiple-exits/cf097-loop-with-multiple-exits_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_multiple_exitsi>:
100000360: 52800009    	mov	w9, #0x0                ; =0
100000364: 529999aa    	mov	w10, #0xcccd            ; =52429
100000368: 72b9998a    	movk	w10, #0xcccc, lsl #16
10000036c: 5293334b    	mov	w11, #0x999a            ; =39322
100000370: 72a3332b    	movk	w11, #0x1999, lsl #16
100000374: 5280064c    	mov	w12, #0x32              ; =50
100000378: 6b09001f    	cmp	w0, w9
10000037c: 54000200    	b.eq	0x1000003bc <__Z19test_multiple_exitsi+0x5c>
100000380: 1b097d28    	mul	w8, w9, w9
100000384: 6b00011f    	cmp	w8, w0
100000388: 540001cc    	b.gt	0x1000003c0 <__Z19test_multiple_exitsi+0x60>
10000038c: 52800008    	mov	w8, #0x0                ; =0
100000390: 12001d2d    	and	w13, w9, #0xff
100000394: 1b0a7dad    	mul	w13, w13, w10
100000398: 138d05ad    	ror	w13, w13, #0x1
10000039c: 6b0b01bf    	cmp	w13, w11
1000003a0: 7a4c3120    	ccmp	w9, w12, #0x0, lo
1000003a4: 540000a8    	b.hi	0x1000003b8 <__Z19test_multiple_exitsi+0x58>
1000003a8: 1100052d    	add	w13, w9, #0x1
1000003ac: 71018d3f    	cmp	w9, #0x63
1000003b0: aa0d03e9    	mov	x9, x13
1000003b4: 54fffe23    	b.lo	0x100000378 <__Z19test_multiple_exitsi+0x18>
1000003b8: aa0803e0    	mov	x0, x8
1000003bc: d65f03c0    	ret
1000003c0: 12800000    	mov	w0, #-0x1               ; =-1
1000003c4: d65f03c0    	ret

00000001000003c8 <_main>:
1000003c8: 12800000    	mov	w0, #-0x1               ; =-1
1000003cc: d65f03c0    	ret
