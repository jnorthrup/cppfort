
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf055-switch-nested/cf055-switch-nested_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z18test_nested_switchii>:
100000360: 52800288    	mov	w8, #0x14               ; =20
100000364: 52800149    	mov	w9, #0xa                ; =10
100000368: 5280018a    	mov	w10, #0xc               ; =12
10000036c: 7100083f    	cmp	w1, #0x2
100000370: 1a890149    	csel	w9, w10, w9, eq
100000374: 5280016a    	mov	w10, #0xb               ; =11
100000378: 7100043f    	cmp	w1, #0x1
10000037c: 1a890149    	csel	w9, w10, w9, eq
100000380: 7100041f    	cmp	w0, #0x1
100000384: 1a8913e9    	csel	w9, wzr, w9, ne
100000388: 7100081f    	cmp	w0, #0x2
10000038c: 1a890100    	csel	w0, w8, w9, eq
100000390: d65f03c0    	ret

0000000100000394 <_main>:
100000394: 52800180    	mov	w0, #0xc                ; =12
100000398: d65f03c0    	ret
