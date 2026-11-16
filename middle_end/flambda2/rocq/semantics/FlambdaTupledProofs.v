(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)


(* Proofs of the statements of FlambdaTupledSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Arith Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaCompareProofs
  FlambdaApplyProofs FlambdaPartialApplicationProofs FlambdaWellFormedSpec
  FlambdaWellKindedSpec FlambdaTupledSpec.
Import ListNotations.

(* ======================================================================== *)
(* Tuples                                                                   *)
(* ======================================================================== *)

Lemma skipn_nth_error {A : Type} (l : list A) :
  ∀ (n : nat) (a : A),
    nth_error l n = Some a -> skipn n l = a :: skipn (S n) l.
Proof.
  induction l as [ | b l IH]; intros [ | n] a; simpl; try discriminate.
  - intros [= <-]. reflexivity.
  - intros Hnth. exact (IH n a Hnth).
Qed.

(* The projections of a block of tag 0 with [size] fields of kind
   [Value] read its fields. *)
Lemma tuple_fields_from_block (statics : static_env) (size : nat)
    (tuple : value) (block : list value) :
  block_of statics tuple = Some (0%Z, block) ->
  length block = size ->
  Forall (fun v => kind_of_value v = Value) block ->
  ∀ count field : nat,
    field + count <= size ->
    tuple_fields_from statics size tuple field count
    = Some (firstn count (skipn field block)).
Proof.
  intros Hblock Hlength Hkinds.
  induction count as [ | count IH]; intros field Hle; [ reflexivity | ].
  cbn [tuple_fields_from eval_prim eval_unary_primitive].
  unfold eval_block_load, tuple_access. rewrite Hblock. cbn [known_is].
  rewrite Hlength, !Z.eqb_refl.
  replace (0 <=? Z.of_nat field)%Z with true
    by (symmetry; apply Z.leb_le; lia).
  cbn [andb]. rewrite Nat2Z.id.
  destruct (nth_error block field) as [v | ] eqn:Hnth.
  2: { apply nth_error_None in Hnth. lia. }
  assert (Hv : kind_of_value v = Value).
  { rewrite Forall_forall in Hkinds. apply Hkinds.
    exact (nth_error_In _ _ Hnth). }
  rewrite Hv. cbn [kind_eqb]. rewrite (IH (S field)) by lia.
  rewrite (skipn_nth_error block field v Hnth). reflexivity.
Qed.

Theorem tuple_is_block : tuple_is_block_statement.
Proof.
  intros statics size tuple fields Hsize. split.
  - intros Hfields.
    pose proof (tuple_fields_length _ _ _ _ Hfields) as Hlength.
    pose proof (tuple_fields_kinds _ _ _ _ Hfields) as Hkinds.
    split; [ | split; [ exact Hlength | ] ].
    + destruct (tuple_fields_block _ _ _ _ Hfields) as [-> | Hblock];
        [ simpl in Hlength; destruct (Hsize (eq_sym Hlength)) | exact Hblock ].
    + clear Hfields Hlength. revert size Hsize Hkinds.
      induction fields as [ | v fields IH]; intros size Hsize Hkinds;
        [ apply Forall_nil | ].
      destruct size as [ | size]; [ discriminate | ].
      injection Hkinds as Hv Hrest. apply Forall_cons; [ exact Hv | ].
      destruct fields as [ | w fields]; [ apply Forall_nil | ].
      apply (IH size); [ | exact Hrest ].
      destruct size; [ discriminate | discriminate ].
  - intros (Hblock & Hlength & Hkinds). unfold tuple_fields.
    rewrite (tuple_fields_from_block statics size tuple fields Hblock Hlength
               Hkinds size 0 (Nat.le_refl _)).
    cbn [skipn]. rewrite <- Hlength, firstn_all. reflexivity.
Qed.

(* ======================================================================== *)
(* Calls of tupled code                                                     *)
(* ======================================================================== *)

(* The call of a closure of tupled code whose first argument is a tuple
   is what the call of the code is with the fields of the tuple
   followed by the other arguments. *)
Lemma shape_of_closure_call_tupled (statics : static_env) (callee : value)
    (kind : call_kind) (tuple : value) (rest : list value)
    (first : arity_component) (rest_arity return_arity : arity)
    (called : code_id) (slots : list (value_slot * value)) (c : code)
    (fields : list value) :
  closure_of statics callee = Some (called, slots) ->
  kind = Indirect ∨ kind = Direct called ->
  lookup called (static_code statics) = Some c ->
  code_is_tupled c = true ->
  map kind_of_value (tuple :: rest) = unarize (first :: rest_arity) ->
  unarize_component first = [Value] ->
  tuple_fields statics (cardinal_unarized (code_params_arity c)) tuple
  = Some fields ->
  shape_of_closure_call statics callee kind (tuple :: rest)
    (first :: rest_arity) return_arity
  = shape_of_code_call callee c (fields ++ rest)
      (arity_of_kinds
         (repeat Value (cardinal_unarized (code_params_arity c)))
       ++ rest_arity)
      return_arity.
Proof.
  intros Hclosure Hkind Hcode Htupled Hkinds Hfirst Hfields.
  unfold shape_of_closure_call. rewrite Hclosure.
  assert (Hexpected :
    match kind with
    | Direct named => Pos.eqb called named
    | Indirect => true
    end = true)
    by (destruct Hkind as [-> | ->]; [ reflexivity | apply Pos.eqb_refl ]).
  rewrite Hexpected, Hcode. unfold arguments_for_code, detuple.
  rewrite Htupled, Hkinds, Hfirst, !kinds_eqb_refl. cbn [andb negb].
  rewrite Hfields. reflexivity.
Qed.

(* Code whose parameters are single values has as many of them,
   unarized, as it has variables. *)
Lemma params_are_values_cardinal (c : code) :
  params_are_values c ->
  cardinal_unarized (code_params_arity c) = length (code_params c)
  ∧ length (code_params_arity c) = length (code_params c).
Proof.
  unfold params_are_values, cardinal_unarized. intros ->.
  rewrite unarize_arity_of_kinds. unfold arity_of_kinds.
  rewrite length_map, repeat_length. split; reflexivity.
Qed.

(* A value that has the fields of a tuple of a size that is not zero is
   of kind [Value]. *)
Lemma tuple_kind (statics : static_env) (size : nat) (tuple : value)
    (fields : list value) :
  size <> 0 ->
  tuple_fields statics size tuple = Some fields ->
  kind_of_value tuple = Value.
Proof.
  intros Hsize Hfields.
  destruct (proj1 (tuple_is_block statics size tuple fields Hsize) Hfields)
    as (Hblock & _).
  apply (block_of_kind statics). rewrite Hblock. reflexivity.
Qed.

Lemma params_not_none (c : code) :
  code_params c <> [] -> length (code_params c) <> 0.
Proof. destruct (code_params c); [ congruence | discriminate ]. Qed.

Theorem apply_tupled : apply_tupled_statement.
Proof.
  intros ap vars conts statics hp rd callee tuple fields extra called slots c
    Hcallee Hargs Hextra Hclosure Hkind Hcode Htupled Hparams Hsome Harity
    Hfields.
  destruct (params_are_values_cardinal c Hparams) as [Hcardinal Hcount].
  rewrite step_apply, Hcallee, Hargs, Hextra. unfold call_function.
  rewrite (shape_of_call_closure statics callee _ _ _ _
             (closure_not_partial statics callee _ Hclosure)).
  rewrite Harity,
    (shape_of_closure_call_tupled statics callee _ tuple [] _ [] _ called
       slots c fields Hclosure Hkind Hcode Htupled);
    [ | simpl;
        rewrite (tuple_kind statics _ tuple fields (params_not_none c Hsome)
                   Hfields);
        reflexivity
    | reflexivity
    | rewrite Hcardinal; exact Hfields ].
  rewrite !app_nil_r, Hcardinal, <- Hparams. unfold shape_of_code_call.
  rewrite Nat.eqb_refl. reflexivity.
Qed.

Theorem apply_tupled_over : apply_tupled_over_statement.
Proof.
  intros ap vars conts statics hp rd callee tuple fields rest extra rest_arity
    called slots c Hcallee Hargs Hextra Hclosure Hkind Hcode Htupled Hparams
    Hsome Harity Hnonempty Hkinds Htest Hfields.
  destruct (params_are_values_cardinal c Hparams) as [Hcardinal Hcount].
  pose proof (tuple_kind statics _ tuple fields (params_not_none c Hsome)
                Hfields) as Htuple.
  rewrite step_apply, Hcallee, Hargs, Hextra. unfold call_function.
  rewrite (shape_of_call_closure statics callee _ _ _ _
             (closure_not_partial statics callee _ Hclosure)).
  rewrite Harity,
    (shape_of_closure_call_tupled statics callee _ tuple rest _ rest_arity _
       called slots c fields Hclosure Hkind Hcode Htupled);
    [ | simpl; rewrite Htuple, Hkinds; reflexivity
    | reflexivity
    | rewrite Hcardinal; exact Hfields ].
  rewrite Hcardinal, <- Hparams. unfold shape_of_code_call.
  assert (Hlonger :
    length (code_params_arity c)
    < length (code_params_arity c ++ rest_arity)).
  { rewrite length_app.
    destruct rest_arity; [ destruct (Hnonempty eq_refl) | simpl; lia ]. }
  assert (Hnot_full :
    Nat.eqb (length (code_params_arity c ++ rest_arity))
      (length (code_params_arity c))
    = false)
    by (apply Nat.eqb_neq; lia).
  rewrite Hnot_full.
  assert (Hall :
    map kind_of_value (fields ++ rest)
    = unarize (code_params_arity c ++ rest_arity)).
  { rewrite map_app, (tuple_fields_kinds _ _ _ _ Hfields), Hkinds.
    unfold unarize. rewrite flat_map_app. f_equal.
    change (flat_map unarize_component (code_params_arity c))
      with (unarize (code_params_arity c)).
    unfold params_are_values in Hparams.
    rewrite Hparams, unarize_arity_of_kinds. reflexivity. }
  rewrite Hall, kinds_eqb_refl, Htest. cbn [andb negb].
  apply Nat.ltb_lt in Hlonger. rewrite Hlonger. cbn [run_call].
  unfold partially_apply.
  rewrite Hcardinal, <- (tuple_fields_length _ _ _ _ Hfields),
    firstn_app_exact, skipn_app_exact, firstn_app_exact, skipn_app_exact.
  reflexivity.
Qed.

Theorem apply_tupled_undefined : apply_tupled_undefined_statement.
Proof.
  intros ap vars conts statics hp rd callee vs called slots c Hcallee Hargs
    Hclosure Hcode Htupled Hwrong.
  rewrite step_apply, Hcallee, Hargs.
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
    [ | reflexivity ].
  unfold call_function.
  rewrite (shape_of_call_closure statics callee _ _ _ _
             (closure_not_partial statics callee _ Hclosure)).
  destruct
    (shape_of_closure_call statics callee (ap_call_kind ap) vs
       (ap_args_arity ap) (ap_return_arity ap))
    as [shape | ] eqn:Hshape;
    [ exfalso | reflexivity ].
  apply shape_of_closure_call_cases in Hshape
    as (called' & slots' & c' & vs' & args_arity' & Hclosure' & Hcode' & _
        & Hdetupled & _).
  rewrite Hclosure in Hclosure'. injection Hclosure' as <- <-.
  rewrite Hcode in Hcode'. injection Hcode' as <-.
  unfold arguments_for_code in Hdetupled. rewrite Htupled in Hdetupled.
  apply detuple_cases in Hdetupled
    as (tuple & rest & first & rest_arity & fields & Hvs & Harity & _
        & Hfirst & Hfields & _).
  destruct Hwrong as [-> | [Hno_value | (tuple' & rest' & -> & Hnone)]].
  - discriminate.
  - exact (Hno_value first rest_arity Harity Hfirst).
  - injection Hvs as <- <-. rewrite Hfields in Hnone. discriminate.
Qed.

Theorem tupled_never_partial : tupled_never_partial_statement.
Proof.
  intros statics closure kind vs args_arity return_arity called slots c
    partial Hclosure Hcode Htupled Hshape.
  apply shape_of_closure_call_cases in Hshape
    as (called' & slots' & c' & vs' & args_arity' & Hclosure' & Hcode' & _
        & _ & Hshape).
  rewrite Hclosure in Hclosure'. injection Hclosure' as <- <-.
  rewrite Hcode in Hcode'. injection Hcode' as <-.
  apply shape_of_code_call_cases in Hshape
    as [[_ Hfull] | [(_ & _ & _ & Hover) | (_ & _ & _ & Hnot & _)]].
  - discriminate.
  - discriminate.
  - rewrite Htupled in Hnot. discriminate.
Qed.

(* ======================================================================== *)
(* The examples                                                             *)
(* ======================================================================== *)

Theorem examples_run : examples_run_statement.
Proof.
  intros e [<- | [<- | [<- | []]]]; exists 20; eexists; vm_compute; reflexivity.
Qed.

Theorem examples_well_formed : examples_well_formed_statement.
Proof.
  intros e [<- | [<- | [<- | []]]]; split; vm_compute; reflexivity.
Qed.

Theorem wrong_tuple_undefined : wrong_tuple_undefined_statement.
Proof.
  intros e [<- | [<- | []]]; exists 20; vm_compute; reflexivity.
Qed.


(* ======================================================================== *)
(* Inlining a call of tupled code                                           *)
(* ======================================================================== *)

(* A projection of [let_tuple_fields] that has a value by [eval_prim] is
   not of a location: a block read as a value is a block or a symbol
   bound to one. *)
Lemma tuple_load_no_location (statics : static_env) (size field : nat)
    (tv v : value) :
  eval_prim statics
    (Unary (Block_load (tuple_access size) Immutable (Z.of_nat field))) [tv]
  = Some v ->
  ∀ l : location, Value_location l ∉ [tv].
Proof.
  intros Hv l [-> | []]. discriminate Hv.
Qed.

(* The step of one [Let] of [let_tuple_fields], where the value of the
   tuple is no location: the projection is as [eval_prim] has it. *)
Lemma step_let_tuple_field (statics : static_env) (hp : heap)
    (rd : read_record) (size field : nat) (param : variable)
    (tuple : simple) (tv : value) (body : expr) (vars : env)
    (conts : cont_env) :
  eval_simple vars tuple = Some tv ->
  (∀ l : location, Value_location l ∉ [tv]) ->
  step
    (State
       (Let param Value
          (Prim
             (Unary
                (Block_load (tuple_access size) Immutable (Z.of_nat field)))
             [tuple])
          body)
       vars conts statics hp rd)
  = match
      eval_prim statics
        (Unary (Block_load (tuple_access size) Immutable (Z.of_nat field)))
        [tv]
    with
    | Some v =>
      if kind_eqb (kind_of_value v) Value
      then Step (State body ((param, v) :: vars) conts statics hp rd)
      else Undefined
    | None => Undefined
    end.
Proof.
  intros Htuple Hnone.
  apply step_let_no_location; [ intros q; discriminate | | exact Hnone ].
  cbn [eval_simples]. rewrite Htuple. reflexivity.
Qed.

(* The [Let]s of [let_tuple_fields] run to the body, with the parameters
   bound to the fields of the tuple as a call binds them, provided no
   parameter is the tuple. *)
Lemma let_tuple_fields_run (statics : static_env) (hp : heap) (rd : read_record)
    (size : nat)
    (tuple : simple) (tv : value) (body : expr) (conts : cont_env) :
  ∀ (params : list variable) (field : nat) (fields : list value)
    (vars bound : env),
    eval_simple vars tuple = Some tv ->
    (∀ param : variable,
      param ∈ params -> ~ var_free_in_simple param tuple) ->
    tuple_fields_from statics size tv field (length params) = Some fields ->
    bind_params params fields vars = Some bound ->
    ∃ n : nat,
      run n
        (State (let_tuple_fields params size field tuple body) vars conts
           statics hp rd)
      = Step (State body bound conts statics hp rd).
Proof.
  induction params as [ | param params IH];
    intros field fields vars bound Htuple Hfresh Hfields Hbind.
  - simpl in Hfields. injection Hfields as <-. simpl in Hbind.
    injection Hbind as <-. exists 0. reflexivity.
  - cbn [length tuple_fields_from] in Hfields.
    destruct
      (eval_prim statics
         (Unary (Block_load (tuple_access size) Immutable (Z.of_nat field)))
         [tv])
      as [v | ] eqn:Hv; [ | discriminate ].
    destruct (tuple_fields_from statics size tv (S field) (length params))
      as [vs | ] eqn:Hvs; [ | discriminate ].
    simpl in Hfields. injection Hfields as <-. simpl in Hbind.
    destruct (IH (S field) vs ((param, v) :: vars) bound) as [n Hrun].
    + rewrite eval_simple_extend; [ exact Htuple | ].
      apply Hfresh. left. reflexivity.
    + intros p Hin. apply Hfresh. right. exact Hin.
    + exact Hvs.
    + exact Hbind.
    + exists (S n). cbn [let_tuple_fields].
      rewrite run_succ,
        (step_let_tuple_field statics hp rd size field param tuple tv _ vars
           conts Htuple (tuple_load_no_location statics size field tv v Hv)).
      rewrite Hv, (proj2 (eval_prim_kinds _ _ _ _ Hv)). simpl.
      exact Hrun.
Qed.

(* Where the tuple has not the fields and is no location, one of the
   [Let]s is undefined. (Of a location a projection reads the cell, and
   may have a value where [tuple_fields_from], which reads none, has
   not.) *)
Lemma let_tuple_fields_wrong (statics : static_env) (hp : heap)
    (rd : read_record) (size : nat)
    (tuple : simple) (tv : value) (body : expr) (conts : cont_env) :
  (∀ l : location, tv <> Value_location l) ->
  ∀ (params : list variable) (field : nat) (vars : env),
    eval_simple vars tuple = Some tv ->
    (∀ param : variable,
      param ∈ params -> ~ var_free_in_simple param tuple) ->
    tuple_fields_from statics size tv field (length params) = None ->
    behaves
      (State (let_tuple_fields params size field tuple body) vars conts
         statics hp rd)
      Goes_wrong.
Proof.
  intros Hvalue.
  assert (Hnone : ∀ l : location, Value_location l ∉ [tv]).
  { intros l [Hl | []]. exact (Hvalue l Hl). }
  induction params as [ | param params IH];
    intros field vars Htuple Hfresh Hfields; [ discriminate | ].
  cbn [length tuple_fields_from] in Hfields. cbn [let_tuple_fields].
  destruct
    (eval_prim statics
       (Unary (Block_load (tuple_access size) Immutable (Z.of_nat field)))
       [tv])
    as [v | ] eqn:Hv.
  - destruct (tuple_fields_from statics size tv (S field) (length params))
      as [vs | ] eqn:Hvs; [ discriminate | ].
    apply (step_behaves _ Goes_wrong).
    rewrite
      (step_let_tuple_field statics hp rd size field param tuple tv _ vars
         conts Htuple Hnone).
    rewrite Hv, (proj2 (eval_prim_kinds _ _ _ _ Hv)). simpl.
    apply (IH (S field)); [ | | exact Hvs ].
    + rewrite eval_simple_extend; [ exact Htuple | ].
      apply Hfresh. left. reflexivity.
    + intros p Hin. apply Hfresh. right. exact Hin.
  - apply step_undefined_wrong.
    rewrite
      (step_let_tuple_field statics hp rd size field param tuple tv _ vars
         conts Htuple Hnone).
    rewrite Hv. reflexivity.
Qed.

(* The first step of the inlined body binds [my_closure]. *)
Lemma inlined_tupled_body_step (c : code) (r : renaming) (ap : apply_expr)
    (tuple : simple) (callee : value) (called : code_id)
    (slots : list (value_slot * value)) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  eval_simple vars (ap_callee ap) = Some callee ->
  closure_of statics callee = Some (called, slots) ->
  step (State (inlined_tupled_body c r ap tuple) vars conts statics hp rd)
  = Step
      (State
         (let_tuple_fields (code_params c) (length (code_params c)) 0 tuple
            (rename_expr r (code_body c)))
         ((code_my_closure c, callee) :: vars) conts statics hp rd).
Proof.
  intros Hcallee Hclosure. unfold inlined_tupled_body. rewrite step_let_simple.
  simpl. rewrite Hcallee, (closure_of_kind _ _ _ _ Hclosure). reflexivity.
Qed.

(* Where the tuple has the fields and the call has stepped to the body
   under the callee's variables, the parameters bound to the fields: the
   inlined body runs its [Let]s to the renamed body under those
   variables followed by the caller's. The two are a state and its
   renaming, as for FlambdaApplyProofs.inline. *)
Lemma inline_tupled_meet (ap : apply_expr) (called : code_id) (c : code)
    (r : renaming) (tuple : simple) (callee tv : value)
    (slots : list (value_slot * value)) (fields : list value)
    (callee_vars vars : env) (conts : cont_env) (statics : static_env)
      (hp : heap) (rd : read_record) :
  tupled_inlinable ap c r tuple ->
  eval_simple vars (ap_callee ap) = Some callee ->
  closure_of statics callee = Some (called, slots) ->
  eval_simple vars tuple = Some tv ->
  tuple_fields statics (length (code_params c)) tv = Some fields ->
  bind_params (code_params c) fields [(code_my_closure c, callee)]
  = Some callee_vars ->
  equivalent
    (State (code_body c) callee_vars
       [(code_return_continuation c, resolve (ap_continuation ap) conts);
        (code_exn_continuation c, resolve (ap_exn_continuation ap) conts)]
       statics hp rd)
    (State (inlined_tupled_body c r ap tuple) vars conts statics hp rd).
Proof.
  intros
    (Hclosed_vars & Hclosed_conts & Hvars & Hret & Hexn & Hextra & Hargs_eq
     & Harity & Hmy_closure & Hparams & Htupled & Hvalues & Hsome
     & Hstatic)
    Hcallee Hclosure Htuple Hfields Hbind.
  destruct
    (let_tuple_fields_run statics hp rd (length (code_params c)) tuple tv
       (rename_expr r (code_body c)) conts (code_params c) 0 fields
       ((code_my_closure c, callee) :: vars) (callee_vars ++ vars))
    as [n Hrun].
  { rewrite eval_simple_extend; assumption. }
  { exact Hparams. }
  { exact Hfields. }
  { exact (bind_params_app (code_params c) fields
      [(code_my_closure c, callee)] callee_vars vars Hbind). }
  apply equivalent_sym.
  eapply equivalent_trans.
  { apply (equivalent_of_run (S n) 0).
    rewrite run_succ,
      (inlined_tupled_body_step c r ap tuple callee called slots vars conts
         statics _ rd_ Hcallee Hclosure).
    exact Hrun. }
  apply equivalent_sym, renamed_equivalent.
  - apply binds_statics_code_closed, Hstatic.
  - intros x Hx. rewrite Hvars, lookup_app.
    destruct (bind_params_bound (code_params c) fields
                [(code_my_closure c, callee)] callee_vars x Hbind)
      as [v Hlookup]; [ | rewrite Hlookup; reflexivity ].
    destruct (Hclosed_vars x Hx) as [-> | Hin]; [ right | left; exact Hin ].
    exists callee. simpl. rewrite Pos.eqb_refl. reflexivity.
  - intros k Hk. unfold conts_agree_at.
    destruct (Hclosed_conts k Hk) as [-> | ->].
    + rewrite Hret. unfold resolve at 1. simpl. rewrite Pos.eqb_refl.
      apply Same_matches.
    + rewrite Hexn. unfold resolve at 1. simpl.
      destruct (Pos.eqb_spec (code_exn_continuation c)
                  (code_return_continuation c)) as [Heq | _].
      * rewrite <- Hexn, Heq, Hret. apply Same_matches.
      * rewrite Pos.eqb_refl. apply Same_matches.
Qed.

(* The call goes wrong, or it steps to the body under the callee's
   variables, where the two meet. *)
Theorem inline_tupled : inline_tupled_statement.
Proof.
  intros ap called c r tuple vars conts statics hp rd Hkind Hcode Hinlinable.
  pose proof Hinlinable as
    (_ & _ & _ & _ & _ & Hextra & Hargs_eq & Harity & _ & _ & Htupled
     & Hvalues & Hsome & Hstatic).
  pose proof
    (step_undefined_refines (State (Apply ap) vars conts statics hp rd)
       (State (inlined_tupled_body c r ap tuple) vars conts statics hp rd))
    as Hwrong.
  destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
    [ | apply Hwrong;
        rewrite (step_apply_no_extra_args _ _ _ _ _ rd_ Hextra), Hcallee;
        reflexivity ].
  destruct (eval_simple vars tuple) as [tv | ] eqn:Htuple;
    [ | apply Hwrong;
        rewrite (step_apply_no_extra_args _ _ _ _ _ rd_ Hextra), Hcallee,
          Hargs_eq;
        simpl; rewrite Htuple; reflexivity ].
  assert (Hargs : eval_simples vars (ap_args ap) = Some [tv])
    by (rewrite Hargs_eq; simpl; rewrite Htuple; reflexivity).
  assert (Hno_extra : eval_simples vars (exn_extra_simples ap) = Some [])
    by (unfold exn_extra_simples; rewrite Hextra; reflexivity).
  destruct (closure_of statics callee) as [[called' slots] | ] eqn:Hclosure;
    [ | apply Hwrong;
        rewrite (step_apply_no_extra_args _ _ _ _ _ rd_ Hextra), Hcallee, Hargs,
          Hkind;
        unfold call_function; rewrite shape_of_call_direct;
        unfold shape_of_closure_call; rewrite Hclosure; reflexivity ].
  destruct (Pos.eqb_spec called' called) as [-> | Hother];
    [ | apply Hwrong;
        rewrite (step_apply_no_extra_args _ _ _ _ _ rd_ Hextra), Hcallee, Hargs,
          Hkind;
        unfold call_function; rewrite shape_of_call_direct;
        unfold shape_of_closure_call; rewrite Hclosure;
        destruct (Pos.eqb_spec called' called);
        [ contradiction | reflexivity ] ].
  destruct (params_are_values_cardinal c Hvalues) as [Hcardinal _].
  destruct (tuple_fields statics (length (code_params c)) tv)
    as [fields | ] eqn:Hfields.
  2: { apply Hwrong.
       apply
         (apply_tupled_undefined ap vars conts statics _ rd_ callee [tv] called
                slots c Hcallee Hargs Hclosure Hcode Htupled).
       right. right. exists tv, []. split; [ reflexivity | ].
       rewrite Hcardinal. exact Hfields. }
  pose proof
    (apply_tupled ap vars conts statics hp rd callee tv fields [] called slots c
       Hcallee Hargs Hno_extra Hclosure (or_intror Hkind) Hcode Htupled
       Hvalues Hsome Harity Hfields)
    as Hstep.
  unfold call_code in Hstep. cbn [with_extra_args] in Hstep.
  rewrite Hstatic in Hstep.
  match type of Hstep with
  | context [negb ?agrees] => destruct agrees
  end;
    [ | exact (Hwrong Hstep) ].
  cbn [negb] in Hstep.
  destruct
    (bind_params (code_params c) fields [(code_my_closure c, callee)])
    as [callee_vars | ] eqn:Hbind; [ | exact (Hwrong Hstep) ].
  apply equivalent_refines.
  eapply equivalent_trans; [ exact (step_equivalent _ _ Hstep) | ].
  exact (inline_tupled_meet ap called c r tuple callee tv slots fields
           callee_vars vars conts statics _ rd_ Hinlinable Hcallee Hclosure
             Htuple
           Hfields Hbind).
Qed.

(* The three cases: the tuple has the fields, it has not, the argument
   has no value. *)
Theorem inline_tupled_equivalent : inline_tupled_equivalent_statement.
Proof.
  intros ap called c r tuple callee slots vars conts statics hp rd Hkind
    Hcallee Hclosure Hcode Hinlinable Hreturn Hvalue.
  pose proof Hinlinable as
    (_ & _ & _ & _ & _ & Hextra & Hargs_eq & Harity & Hmy_closure & Hparams
     & Htupled & Hvalues & Hsome & Hstatic).
  pose proof
    (inlined_tupled_body_step c r ap tuple callee called slots vars conts
       statics hp rd Hcallee Hclosure)
    as Hfirst.
  destruct (params_are_values_cardinal c Hvalues) as [Hcardinal _].
  destruct (eval_simple vars tuple) as [tv | ] eqn:Htuple.
  2: { apply both_wrong_equivalent.
       - apply step_undefined_wrong.
         rewrite (step_apply_no_extra_args _ _ _ _ _ rd_ Hextra), Hcallee,
           Hargs_eq.
         simpl. rewrite Htuple. reflexivity.
       - apply (step_behaves _ Goes_wrong). rewrite Hfirst.
         destruct (code_params c) as [ | param params];
           [ contradiction Hsome; reflexivity | ].
         apply step_undefined_wrong. cbn [let_tuple_fields].
         rewrite step_let_exec, exec_named_no_args; [ reflexivity | ].
         cbn [eval_simples].
         rewrite eval_simple_extend by exact Hmy_closure.
         rewrite Htuple. reflexivity. }
  assert (Hargs : eval_simples vars (ap_args ap) = Some [tv])
    by (rewrite Hargs_eq; simpl; rewrite Htuple; reflexivity).
  assert (Hno_extra : eval_simples vars (exn_extra_simples ap) = Some [])
    by (unfold exn_extra_simples; rewrite Hextra; reflexivity).
  destruct (tuple_fields statics (length (code_params c)) tv)
    as [fields | ] eqn:Hfields.
  2: { apply both_wrong_equivalent.
       - apply step_undefined_wrong.
         apply (apply_tupled_undefined ap vars conts statics _ rd_ callee
                  [tv] called slots c Hcallee Hargs Hclosure Hcode Htupled).
         right. right. exists tv, []. split; [ reflexivity | ].
         rewrite Hcardinal. exact Hfields.
       - apply (step_behaves _ Goes_wrong). rewrite Hfirst.
         apply (let_tuple_fields_wrong statics _ rd_ _ tuple tv);
           [ intros l ->; exact (Hvalue l eq_refl)
           | rewrite eval_simple_extend; assumption | exact Hparams
           | exact Hfields ]. }
  pose proof
    (apply_tupled ap vars conts statics hp rd callee tv fields [] called
       slots c Hcallee Hargs Hno_extra Hclosure (or_intror Hkind) Hcode
       Htupled Hvalues Hsome Harity Hfields)
    as Hstep.
  unfold call_code in Hstep. cbn [with_extra_args] in Hstep.
  rewrite Hstatic, Hreturn, arity_eqb_refl,
    (proj2 (kinds_per_parameter_eqb_eq _ _) eq_refl),
    (tuple_fields_kinds _ _ _ _ Hfields)
    in Hstep.
  unfold params_are_values in Hvalues.
  rewrite Hvalues, unarize_arity_of_kinds, kinds_eqb_refl in Hstep.
  cbn [andb negb] in Hstep.
  destruct (bind_params_some (code_params c) fields
              [(code_my_closure c, callee)])
    as [callee_vars Hbind].
  { rewrite (tuple_fields_length _ _ _ _ Hfields). reflexivity. }
  rewrite Hbind in Hstep.
  eapply equivalent_trans; [ exact (step_equivalent _ _ Hstep) | ].
  exact (inline_tupled_meet ap called c r tuple callee tv slots fields
           callee_vars vars conts statics _ rd_ Hinlinable Hcallee Hclosure
           Htuple Hfields Hbind).
Qed.
