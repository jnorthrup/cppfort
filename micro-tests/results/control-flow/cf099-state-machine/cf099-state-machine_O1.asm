
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf099-state-machine/cf099-state-machine_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18test_state_machinei>:
100000360: 52800008    	mov	w8, #0x0                ; =0
100000364: 52800009    	mov	w9, #0x0                ; =0
100000368: 531f780a    	lsl	w10, w0, #1
10000036c: 7100041f    	cmp	w0, #0x1
100000370: 5280006b    	mov	w11, #0x3               ; =3
100000374: 1a9fb56c    	csinc	w12, w11, wzr, lt
100000378: 5280004d    	mov	w13, #0x2               ; =2
10000037c: 7100013f    	cmp	w9, #0x0
100000380: 1a8c116e    	csel	w14, w11, w12, ne
100000384: 7100053f    	cmp	w9, #0x1
100000388: 1a8e01a9    	csel	w9, w13, w14, eq
10000038c: 1a880148    	csel	w8, w10, w8, eq
100000390: 321e752e    	orr	w14, w9, #0xfffffffc
100000394: 310009df    	cmn	w14, #0x2
100000398: 54ffff23    	b.lo	0x10000037c <__Z18test_state_machinei+0x1c>
10000039c: 7100093f    	cmp	w9, #0x2
1000003a0: 5a9f0100    	csinv	w0, w8, wzr, eq
1000003a4: d65f03c0    	ret

00000001000003a8 <_main>:
1000003a8: 52800008    	mov	w8, #0x0                ; =0
1000003ac: 52800069    	mov	w9, #0x3                ; =3
1000003b0: 1280000a    	mov	w10, #-0x1              ; =-1
1000003b4: 5280004b    	mov	w11, #0x2               ; =2
1000003b8: 5280054c    	mov	w12, #0x2a              ; =42
1000003bc: 5280002d    	mov	w13, #0x1               ; =1
1000003c0: 2a0a03ee    	mov	w14, w10
1000003c4: 7100011f    	cmp	w8, #0x0
1000003c8: 1a8d112f    	csel	w15, w9, w13, ne
1000003cc: 7100051f    	cmp	w8, #0x1
1000003d0: 1a8e0180    	csel	w0, w12, w14, eq
1000003d4: 1a8f0168    	csel	w8, w11, w15, eq
1000003d8: 321e750e    	orr	w14, w8, #0xfffffffc
1000003dc: 310009df    	cmn	w14, #0x2
1000003e0: 54ffff03    	b.lo	0x1000003c0 <_main+0x18>
1000003e4: d65f03c0    	ret
