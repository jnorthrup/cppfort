
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf097-loop-with-multiple-exits/cf097-loop-with-multiple-exits_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_multiple_exitsi>:
100000360: 52800009    	mov	w9, #0x0                ; =0
100000364: 529999aa    	mov	w10, #0xcccd            ; =52429
100000368: 72b9998a    	movk	w10, #0xcccc, lsl #16
10000036c: 5280064b    	mov	w11, #0x32              ; =50
100000370: 6b09001f    	cmp	w0, w9
100000374: 54000220    	b.eq	0x1000003b8 <__Z19test_multiple_exitsi+0x58>
100000378: 1b097d28    	mul	w8, w9, w9
10000037c: 6b00011f    	cmp	w8, w0
100000380: 540001ec    	b.gt	0x1000003bc <__Z19test_multiple_exitsi+0x5c>
100000384: 52800008    	mov	w8, #0x0                ; =0
100000388: 9baa7d2c    	umull	x12, w9, w10
10000038c: d363fd8c    	lsr	x12, x12, #35
100000390: 0b0c098c    	add	w12, w12, w12, lsl #2
100000394: 531f798c    	lsl	w12, w12, #1
100000398: 6b09019f    	cmp	w12, w9
10000039c: 7a4b0120    	ccmp	w9, w11, #0x0, eq
1000003a0: 540000a8    	b.hi	0x1000003b4 <__Z19test_multiple_exitsi+0x54>
1000003a4: 1100052c    	add	w12, w9, #0x1
1000003a8: 71018d3f    	cmp	w9, #0x63
1000003ac: aa0c03e9    	mov	x9, x12
1000003b0: 54fffe03    	b.lo	0x100000370 <__Z19test_multiple_exitsi+0x10>
1000003b4: aa0803e0    	mov	x0, x8
1000003b8: d65f03c0    	ret
1000003bc: 12800000    	mov	w0, #-0x1               ; =-1
1000003c0: d65f03c0    	ret

00000001000003c4 <_main>:
1000003c4: 12800000    	mov	w0, #-0x1               ; =-1
1000003c8: d65f03c0    	ret
