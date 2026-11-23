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



(* The proofs of FlambdaReaperUnboxingRefinesHeapSpec.v. Nothing here needs
   review.

   The relation of FlambdaReaperUnboxingRefinesProofs.v between the two
   rebuilt terms ([ub]) and its simulation, copied with the primitives
   of the heap ([ubh], [relatedh]): the [Let] of a variable that is not
   unboxed may be of a primitive of the heap, which the two runs make on
   the same values and the same heap; the allocation of an unboxed
   variable reads no heap ([named_reads_heap]); and the two heaps are
   one, not the empty heap. A read of an unboxed variable runs on a
   heap as without one ([read_exec]): its argument is a block or a boxed
   number, or a location, of which the unboxed fields hold no path. The
   rest is that of FlambdaReaperUnboxingRefinesProofs.v, of the graph of
   the units of [in_heap_fragment]. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaNameAbstractionSpec FlambdaRenameApartSpec FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaHeapSpec
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaUnusedStaticsSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionProofs FlambdaReaperRebuild
  FlambdaReaperFlowSpec FlambdaReaperRebuildSpec FlambdaReaperRebuildProofs
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperUnboxingRefinesHeapSpec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperCallingConventionsProofs FlambdaFreshQuantifierSpec
  FlambdaPrimitiveProofs FlambdaNameAbstractionSpec FlambdaRenameApartSpec
  FlambdaHeapProofs.
Import ListNotations.

Module URP := FlambdaReaperUnboxingRefinesProofs.

(* ======================================================================== *)
(* Terms related by local unboxing, with the heap                           *)
(* ======================================================================== *)

Section Related_heap.

Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.

Abbreviation named_plain :=
  (URP.named_plain M).
Abbreviation alloc_entry :=
  (URP.alloc_entry M).
Abbreviation params_rel :=
  (URP.params_rel M).
Abbreviation args_rel :=
  (URP.args_rel M).
Abbreviation ac_rel :=
  (URP.ac_rel M).
Abbreviation env_rel :=
  (URP.env_rel M).
(* [K] are the shapes of the parameters of the continuations in scope. *)
Inductive ubh : finite_map (list (option unboxed_fields)) -> expr -> expr
    -> Prop :=
  | ubh_let (K : finite_map (list (option unboxed_fields))) (x : variable)
      (kind : flambda_kind) (d : named) (body body' : expr) :
    plain_variable M x = true -> named_plain d = true ->
    ubh K body body' ->
    ubh K (Let x kind d body) (Let x kind d body')
  | ubh_alloc (K : finite_map (list (option unboxed_fields))) (x : variable)
      (d : named) (fields : unboxed_fields)
      (ls : list (list (variable * flambda_kind * simple)))
      (body body' : expr) :
    fields_of M x = Some fields ->
    among x (field_variables M) = false ->
    Forall2 (alloc_entry x d) fields ls ->
    named_reads_heap d = false ->
    ubh K body body' ->
    ubh K (Let x Value d body) (lets (concat ls) body')
  | ubh_alias (K : finite_map (list (option unboxed_fields))) (x y : variable)
      (fields fields_y : unboxed_fields)
      (pairs : list (variable * variable * flambda_kind))
      (body body' : expr) :
    fields_of M x = Some fields ->
    among x (field_variables M) = false ->
    fields_of M y = Some fields_y -> x <> y ->
    unboxed_field_pairs Value (Unboxed fields) (Unboxed fields_y)
    = Some pairs ->
    ubh K body body' ->
    ubh K (Let x Value (Simple (Var y)) body) (lets (pairs_lets pairs) body')
  | ubh_load (K : finite_map (list (option unboxed_fields))) (x : variable)
      (kind : flambda_kind) (d : named) (f : field) (s : simple)
      (y : variable) (fields fields_y nested : unboxed_fields)
      (pairs : list (variable * variable * flambda_kind))
      (body body' : expr) :
    fields_of M x = Some fields ->
    among x (field_variables M) = false ->
    read_of d = Some (f, s) ->
    (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
    fields_of M y = Some fields_y -> x <> y ->
    (f, Unboxed nested) ∈ fields_y ->
    unboxed_field_pairs Value (Unboxed fields) (Unboxed nested)
    = Some pairs ->
    ubh K body body' ->
    ubh K (Let x Value d body) (lets (pairs_lets pairs) body')
  | ubh_read (K : finite_map (list (option unboxed_fields))) (x : variable)
      (kind : flambda_kind) (d : named) (f : field)
      (s : simple) (y v : variable) (fields : unboxed_fields)
      (body body' : expr) :
    plain_variable M x = true ->
    read_of d = Some (f, s) ->
    (s = Var y ∨ ∃ k : flambda_kind, s = Const (Poison k)) ->
    fields_of M y = Some fields ->
    (f, Not_unboxed v) ∈ fields ->
    ubh K body body' ->
    ubh K (Let x kind d body) (Let x kind (Simple (Var v)) body')
  | ubh_static (K : finite_map (list (option unboxed_fields)))
      (bindings : list static_binding) (body body' : expr) :
    forallb (local_binding M) bindings = true -> ubh K body body' ->
    ubh K (Let_static bindings body) (Let_static bindings body')
  | ubh_let_cont (K : finite_map (list (option unboxed_fields)))
      (k : continuation) (sh : list (option unboxed_fields))
      (ps ps' : list (variable * flambda_kind))
      (metadata metadata' : handler_metadata)
      (handler handler' body body' : expr) :
    params_rel sh ps ps' -> hm_params_kinds metadata = map snd ps ->
    hm_params_kinds metadata' = map snd ps' -> ubh K handler handler' ->
    ubh ((k, sh) :: K) body body' ->
    ubh K
      (Let_cont_non_recursive k (Handler (map fst ps) metadata handler) body)
      (Let_cont_non_recursive k (Handler (map fst ps') metadata' handler')
         body')
  | ubh_let_cont_rec (K S : finite_map (list (option unboxed_fields)))
      (handlers handlers' : list (continuation * continuation_handler))
      (body body' : expr) :
    ubh_handlers (S ++ K) handlers handlers' S ->
    handlers_bind_statics handlers = handlers_bind_statics handlers' ->
    ubh (S ++ K) body body' ->
    ubh K (Let_cont_recursive handlers body)
      (Let_cont_recursive handlers' body')
  | ubh_apply_cont (K : finite_map (list (option unboxed_fields)))
      (ac ac' : apply_cont) :
    ac_rel K ac ac' -> ubh K (Apply_cont ac) (Apply_cont ac')
  | ubh_switch (K : finite_map (list (option unboxed_fields)))
      (scrutinee : simple) (arms arms' : list (Z * apply_cont)) :
    plain_simple M scrutinee = true ->
    Forall2
      (fun (arm arm' : Z * apply_cont) =>
         fst arm' = fst arm ∧ ac_rel K (snd arm) (snd arm'))
      arms arms' ->
    ubh K (Switch scrutinee arms) (Switch scrutinee arms')
  | ubh_invalid (K : finite_map (list (option unboxed_fields))) :
    ubh K Invalid Invalid
with ubh_handlers : finite_map (list (option unboxed_fields))
    -> list (continuation * continuation_handler)
    -> list (continuation * continuation_handler)
    -> finite_map (list (option unboxed_fields)) -> Prop :=
  | ubh_handlers_nil (K : finite_map (list (option unboxed_fields))) :
    ubh_handlers K [] [] []
  | ubh_handlers_cons (K : finite_map (list (option unboxed_fields)))
      (k : continuation) (sh : list (option unboxed_fields))
      (ps ps' : list (variable * flambda_kind))
      (metadata metadata' : handler_metadata) (handler handler' : expr)
      (handlers handlers' : list (continuation * continuation_handler))
      (S : finite_map (list (option unboxed_fields))) :
    params_rel sh ps ps' -> hm_params_kinds metadata = map snd ps ->
    hm_params_kinds metadata' = map snd ps' -> ubh K handler handler' ->
    ubh_handlers K handlers handlers' S ->
    ubh_handlers K ((k, Handler (map fst ps) metadata handler) :: handlers)
      ((k, Handler (map fst ps') metadata' handler') :: handlers')
      ((k, sh) :: S).


Inductive cont_relh : list (option unboxed_fields) -> cont_value -> cont_value
    -> Prop :=
  | cont_relh_closure (sh : list (option unboxed_fields))
      (ps ps' : list (variable * flambda_kind))
      (metadata metadata' : handler_metadata) (handler handler' : expr)
      (vars vars' : env) (K S : finite_map (list (option unboxed_fields)))
      (conts conts' : cont_env)
      (recursive recursive' : list (continuation * continuation_handler)) :
    params_rel sh ps ps' -> hm_params_kinds metadata = map snd ps ->
    hm_params_kinds metadata' = map snd ps' -> ubh (S ++ K) handler handler' ->
    env_rel vars vars' -> conts_relh K conts conts' ->
    ubh_handlers (S ++ K) recursive recursive' S ->
    cont_relh sh
      (Closure (Handler (map fst ps) metadata handler) vars conts recursive)
      (Closure (Handler (map fst ps') metadata' handler') vars' conts'
         recursive')
with conts_relh : finite_map (list (option unboxed_fields)) -> cont_env
    -> cont_env -> Prop :=
  | conts_relh_nil : conts_relh [] [] []
  | conts_relh_cons (k : continuation) (sh : list (option unboxed_fields))
      (c c' : cont_value) (K : finite_map (list (option unboxed_fields)))
      (conts conts' : cont_env) :
    cont_relh sh c c' -> conts_relh K conts conts' ->
    conts_relh ((k, sh) :: K) ((k, c) :: conts) ((k, c') :: conts').

Definition relatedh (s t : state) : Prop :=
  ∃ K : finite_map (list (option unboxed_fields)),
    ubh K (state_expr s) (state_expr t) ∧ env_rel (state_vars s) (state_vars t)
    ∧ conts_relh K (state_conts s) (state_conts t)
    ∧ state_statics s = state_statics t ∧ state_heap s = state_heap t
    ∧ state_reads s = state_reads t.


End Related_heap.

Scheme ubh_mut := Induction for ubh Sort Prop
  with ubh_handlers_mut := Induction for ubh_handlers Sort Prop.

Section Envs_heap.

Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.

Abbreviation plain_simple_eval :=
  (URP.plain_simple_eval M).
Abbreviation plain_simples_eval :=
  (URP.plain_simples_eval M).
Abbreviation named_plain_eval :=
  (URP.named_plain_eval M).
Abbreviation named_plain_exec :=
  (URP.named_plain_exec M).
Abbreviation local_bindings_eval :=
  (URP.local_bindings_eval M).
Abbreviation env_rel_bind :=
  (URP.env_rel_bind M).
Abbreviation env_rel_empty :=
  (URP.env_rel_empty M).
Lemma resolve_relh (K : finite_map (list (option unboxed_fields)))
    (conts conts' : cont_env) (k : continuation) :
  conts_relh M K conts conts' ->
  (lookup k K = None ∧ resolve k conts = Outer k
   ∧ resolve k conts' = Outer k)
  ∨ ∃ sh : list (option unboxed_fields),
      lookup k K = Some sh
      ∧ cont_relh M sh (resolve k conts) (resolve k conts').
Proof.
  intros Hrel. unfold resolve.
  induction Hrel as [ | k' sh c c' K conts conts' Hc _ IH]; simpl;
    [ left; split; [ reflexivity | split; reflexivity ] | ].
  rewrite ?lookup_cons. destruct (Pos.eqb k k').
  - right. exists sh. split; [ reflexivity | exact Hc ].
  - exact IH.
Qed.

Lemma rec_closures_relh_from
    (KK K S S0 : finite_map (list (option unboxed_fields)))
    (all all' hs hs' : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  KK = S ++ K ->
  ubh_handlers M KK all all' S -> ubh_handlers M KK hs hs' S0 ->
  env_rel M vars vars' -> conts_relh M K conts conts' ->
  conts_relh M (S0 ++ K)
    (map (fun '(k, h) => (k, Closure h vars conts all)) hs ++ conts)
    (map (fun '(k, h) => (k, Closure h vars' conts' all')) hs' ++ conts').
Proof.
  intros HKK Hall Hhs Hvars Hconts.
  induction Hhs as [ | KK0 k sh ps ps' md md' h h' hs hs' S1 Hps Hmd Hmd' Hh
                       _ IH];
    simpl; [ exact Hconts | ].
  constructor; [ | exact (IH HKK Hall) ].
  subst KK0. econstructor; eassumption.
Qed.

Lemma rec_closures_relh (K S : finite_map (list (option unboxed_fields)))
    (handlers handlers' : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  ubh_handlers M (S ++ K) handlers handlers' S -> env_rel M vars vars' ->
  conts_relh M K conts conts' ->
  conts_relh M (S ++ K) (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers' vars' conts' ++ conts').
Proof.
  intros Hh.
  exact (rec_closures_relh_from (S ++ K) K S S _ _ _ _ vars vars' conts conts'
           eq_refl Hh Hh).
Qed.


End Envs_heap.

Section Agree_heap.

Variable M : list (node * unboxed_fields).

Lemma ubh_handlers_keys (K : finite_map (list (option unboxed_fields)))
    (hs hs' : list (continuation * continuation_handler))
    (S : finite_map (list (option unboxed_fields))) :
  ubh_handlers M K hs hs' S -> map fst S = map fst hs.
Proof.
  induction 1 as [ | K k sh ps ps' md md' h h' hs hs' S _ _ _ _ _ IH];
    simpl; [ reflexivity | rewrite IH; reflexivity ].
Qed.


(* The shapes of the continuations not free in a term do not matter. *)
Lemma ubh_agree :
  ∀ (K : finite_map (list (option unboxed_fields))) (p q : expr),
    ubh M K p q ->
    ∀ K' : finite_map (list (option unboxed_fields)),
      agree_on (free_continuations_of_expr p) K K' -> ubh M K' p q.
Proof.
  apply (ubh_mut M
           (fun K p q _ =>
              ∀ K', agree_on (free_continuations_of_expr p) K K' ->
                    ubh M K' p q)
           (fun K hs hs' S _ =>
              ∀ K',
                agree_on
                  (flat_map (fun '(_, h) => free_continuations_of_handler h)
                     hs)
                  K K' ->
                ubh_handlers M K' hs hs' S)).
  - intros K x kind d body body' Hx Hd _ IH K' Hagree.
    apply ubh_let; [ exact Hx | exact Hd | apply IH, Hagree ].
  - intros K x d fields l body body' Hf Hxf Hl Hoff _ IH K' Hagree.
    apply ubh_alloc with (fields := fields); try assumption.
    apply IH, Hagree.
  - intros K x y fields fields_y ls body body' Hf Hxf Hy Hne Hl _ IH K'
      Hagree.
    apply (ubh_alias M K' x y fields fields_y ls); try assumption.
    apply IH, Hagree.
  - intros K x kind d f s y fields fields_y nested l body body' Hf Hxf Hr Hs
      Hy Hne Hin Hl _ IH K' Hagree.
    eapply ubh_load; try eassumption. apply IH, Hagree.
  - intros K x kind d f s y v fields body body' Hx Hr Hs Hf Hin _ IH K'
      Hagree.
    eapply ubh_read; try eassumption. apply IH, Hagree.
  - intros K bindings body body' Hb _ IH K' Hagree.
    apply ubh_static; [ exact Hb | ]. apply IH.
    intros k Hk. apply Hagree. simpl. apply in_or_app. right. exact Hk.
  - intros K k sh ps ps' md md' h h' body body' Hps Hmd Hmd' _ IHh _ IHb K'
      Hagree.
    apply (ubh_let_cont M K' k sh ps ps' md md' h h' body body');
      try assumption.
    + apply IHh. intros k' Hk. apply Hagree. simpl. apply in_or_app.
      left. exact Hk.
    + apply IHb. intros k' Hk. rewrite !lookup_cons.
      destruct (Pos.eqb k' k) eqn:E; [ reflexivity | ].
      apply Hagree. simpl. apply in_or_app. right. apply in_without_iff.
      split; [ exact Hk | ]. intros [Heq | []]. subst k'.
      rewrite Pos.eqb_refl in E. discriminate.
  - intros K S hs hs' body body' Hhs IHhs Hst _ IHb K' Hagree.
    pose proof (ubh_handlers_keys _ _ _ _ Hhs) as Hkeys.
    apply (ubh_let_cont_rec M K' S); [ apply IHhs | exact Hst | apply IHb ];
      apply (agree_app S K K' (map fst hs)); try exact Hkeys;
      intros k Hk; apply Hagree; simpl; apply in_without_iff;
      apply in_without_iff in Hk as [Hk Hnot]; (split; [ | exact Hnot ]);
      apply in_or_app; [ left | right ]; exact Hk.
  - intros K ac ac' (Hk & Htrap & Hargs) K' Hagree.
    apply ubh_apply_cont. split; [ exact Hk | split; [ exact Htrap | ] ].
    unfold shapes_at. rewrite (Hagree (ac_continuation ac) (or_introl eq_refl)).
    exact Hargs.
  - intros K scrutinee arms arms' Hs Harms K' Hagree.
    apply ubh_switch; [ exact Hs | ]. simpl in Hagree. clear Hs.
    induction Harms as [ | [d ac] [d' ac'] arms arms' [Hd (Hk & Htrap & Hargs)]
                         _ IH];
      constructor.
    + split; [ exact Hd | ]. split; [ exact Hk | split; [ exact Htrap | ] ].
      simpl in Hk, Htrap, Hargs |- *. unfold shapes_at in Hargs |- *.
      rewrite (Hagree (ac_continuation ac) (or_introl eq_refl)). exact Hargs.
    + apply IH. intros k Hin. apply Hagree. simpl. right. apply in_or_app.
      right. exact Hin.
  - intros K K' _. apply ubh_invalid.
  - intros K K' _. apply ubh_handlers_nil.
  - intros K k sh ps ps' md md' h h' hs hs' S Hps Hmd Hmd' _ IHh _ IHhs K'
      Hagree.
    apply ubh_handlers_cons; try assumption.
    + apply IHh. intros k' Hk. apply Hagree. simpl. apply in_or_app. left.
      exact Hk.
    + apply IHhs. intros k' Hk. apply Hagree. simpl. apply in_or_app.
      right. exact Hk.
Qed.


End Agree_heap.

(* ======================================================================== *)
(* The simulation, with the heap                                            *)
(* ======================================================================== *)

(* A [Let] that reads no heap runs as [eval_named] says, the heap and the
   record as they came. *)
Lemma exec_named_not_reading (statics : static_env) (hp : heap)
    (rcd : read_record) (vars : env) (d : named) (w : value) (hp1 : heap)
    (rcd1 : read_record) :
  named_reads_heap d = false ->
  exec_named statics hp rcd vars d = Some (w, hp1, rcd1) ->
  eval_named statics vars d = Some w ∧ hp1 = hp ∧ rcd1 = rcd.
Proof.
  intros Hd Hx. destruct d as [s | p args | | | | | ].
  2: exact (FlambdaSemanticsProofs.exec_named_eval_named statics hp hp1 rcd
              rcd1 vars p args w Hd Hx).
  all: cbn [exec_named] in Hx;
       lazymatch type of Hx with
       | context [eval_named ?st ?vs ?dd] =>
         destruct (eval_named st vs dd) as [v | ] eqn:Hev
       end; cbn in Hx; [ | discriminate Hx ];
       injection Hx as <- <- <-; auto.
Qed.

Lemma location_or_not (w : value) :
  (∃ l : location, w = Value_location l)
  ∨ ∀ l : location, w <> Value_location l.
Proof.
  destruct w; try (right; intros l; discriminate). left. eexists. reflexivity.
Qed.

(* A read of a field runs on a heap as [read_eval] says: of a location,
   it has no field to read, and the read may give anything. *)
Lemma read_exec (statics : static_env) (hp : heap) (rcd : read_record)
    (vars : env) (d : named) (f : field) (s : simple) (r : value)
    (hp1 : heap) (rcd1 : read_record) :
  read_of d = Some (f, s) ->
  exec_named statics hp rcd vars d = Some (r, hp1, rcd1) ->
  hp1 = hp ∧ rcd1 = rcd
  ∧ ∃ w0 : value,
      eval_simple vars s = Some w0 ∧ (∀ c : const, w0 <> Value_const c)
      ∧ ∀ w' : value, field_value f w0 = Some w' -> r = w'.
Proof.
  intros Hread Hx.
  pose proof (exec_named_heap statics hp hp1 rcd rcd1 vars d r
                (read_not_heap d f s Hread) Hx) as Hh1.
  subst hp1.
  destruct d as [ | p args | | | | | ]; try discriminate Hread.
  destruct p as [up | | | | ]; try discriminate Hread.
  destruct args as [ | a [ | ]];
    [ destruct up; discriminate Hread | | destruct up; discriminate Hread ].
  cbn [exec_named eval_simples] in Hx.
  destruct (eval_simple vars a) as [w0 | ] eqn:Ha; cbn in Hx;
    [ | discriminate Hx ].
  destruct (exec_prim statics hp (Unary up) [w0]) as [[v hp'] | ] eqn:He;
    cbn in Hx; [ | discriminate Hx ].
  injection Hx as <- -> <-. split; [ reflexivity | split; [ reflexivity | ] ].
  assert (Has : s = a).
  { destruct up; try discriminate Hread; cbn in Hread;
      injection Hread as _ <-; reflexivity. }
  subst a.
  destruct (location_or_not w0) as [(l & ->) | Hnl].
  - exists (Value_location l).
    split; [ exact Ha | split; [ intros c; discriminate | ] ].
    intros w' Hf. exfalso.
    destruct up; try discriminate Hread; cbn in Hread;
      injection Hread; intros; subst; discriminate Hf.
  - rewrite (FlambdaHeapProofs.exec_prim_no_location statics hp (Unary up)
               [w0] ltac:(intros q; discriminate)
               ltac:(intros l [Hl | []]; exact (Hnl l Hl)))
      in He.
    destruct (eval_prim statics (Unary up) [w0]) as [v' | ] eqn:Hev;
      cbn in He; [ | discriminate He ].
    injection He as ->.
    apply (read_eval statics vars _ f s v Hread).
    cbn [eval_named eval_simples]. rewrite Ha. cbn. exact Hev.
Qed.

Section Simulation_heap.

Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.

Abbreviation env_rel_bind_fields :=
  (URP.env_rel_bind_fields M M_distinct).
Abbreviation lets_case :=
  (URP.lets_case M M_distinct).
Abbreviation params_rel_nil_inv :=
  (URP.params_rel_nil_inv M).
Abbreviation params_rel_cons_inv :=
  (URP.params_rel_cons_inv M).
Abbreviation params_rel_none_inv :=
  (URP.params_rel_none_inv M).
Abbreviation params_rel_some_inv :=
  (URP.params_rel_some_inv M).
Abbreviation args_rel_nil_inv :=
  (URP.args_rel_nil_inv M).
Abbreviation alloc_entries_variables :=
  (URP.alloc_entries_variables M).
Abbreviation bind_args_rel :=
  (URP.bind_args_rel M M_distinct).
Abbreviation find_arm_rel :=
  (URP.find_arm_rel M).
Lemma relatedh_answered (s t : state) :
  relatedh M s t -> answered (relatedh M) eq measure s t.
Proof.
  destruct s as [p vars conts statics hp rcd],
    t as [q vars' conts' statics' hp' rcd'].
  intros (K & Hub & Hvars & Hconts & Hstatics & Hhp & Hrcd).
  cbn [state_expr state_vars state_conts state_statics state_heap
       state_reads] in *.
  subst statics' hp' rcd'. unfold answered. revert Hconts.
  destruct Hub as [K x kind d body body' Hx Hd Hbody
    | K x d fields ls body body' Hfields Hxf Hl Hoff Hbody
    | K x y fields fields_y pairs body body' Hfields Hxf Hy Hne Hl Hbody
    | K x kind d f s y fields fields_y nested pairs body body' Hfields Hxf Hread
        Hs Hy Hne Hin Hl Hbody
    | K x kind d f s y v fields body body' Hx Hread Hs Hfields Hin Hbody
    | K bindings body body' Hbindings Hbody
    | K k sh ps ps' md md' h h' body body' Hps Hmd Hmd' Hh Hbody
    | K S handlers handlers' body body' Hhandlers Hstat Hbody
    | K ac ac' Hac
    | K scrutinee arms arms' Hscrutinee Harms
    | K ]; intros Hconts.
  - (* A [Let] of both. *)
    rewrite step_let_exec,
      (named_plain_exec M statics hp rcd vars vars' d Hvars Hd).
    destruct (exec_named statics hp rcd vars' d) as [[[w hp1] rcd1] | ]
      eqn:Hw; [ | exact I ].
    destruct (kind_eqb (kind_of_value w) kind) eqn:Hk; [ | exact I ].
    right. left. eexists.
    split; [ rewrite step_let_exec, Hw, Hk; reflexivity | ].
    exists K. split; [ exact Hbody | ].
    split; [ exact (env_rel_bind M _ _ x w Hvars Hx) | ].
    split; [ exact Hconts | repeat split ].
  - (* The allocation of an unboxed variable. *)
    rewrite step_let_exec.
    destruct (exec_named statics hp rcd vars d) as [[[w hp1] rcd1] | ]
      eqn:Hx; [ | exact I ].
    destruct (exec_named_not_reading statics hp rcd vars d w hp1 rcd1 Hoff Hx)
      as (Hw & -> & ->).
    destruct (kind_eqb (kind_of_value w) Value) eqn:Hk; [ | exact I ].
    destruct (lets_case vars vars' x fields w (concat ls) body' conts' statics
                hp rcd Hvars Hfields)
      as (vars1 & Hrun & Hrel).
    + exact (alloc_entries_variables x d fields ls Hl).
    + intros v kind s v' Hin Hv' ->.
      destruct (forall2_concat_in _ _ _ _ Hl Hin)
        as ([f u] & l0 & _ & Hent & Hin0).
      destruct u as [v0 | nested]; cbn [alloc_entry] in Hent.
      * destruct Hent as (kind0 & s0 & -> & _ & Hplain).
        destruct Hin0 as [Heq | []]. injection Heq as <- <- ->.
        simpl in Hplain. apply plain_variable_iff in Hplain as [_ Hnot].
        pose proof (field_variables_in M x fields v' Hfields Hv') as Hfv.
        apply among_In in Hfv. rewrite Hfv in Hnot. discriminate.
      * destruct Hent as (b & fb & pairs & _ & Hb & Hxb & Hpairs & ->).
        apply in_pairs_lets in Hin0 as (v'' & Heq & Hin').
        injection Heq as <-.
        destruct (pairs_spec _ _ _ _ Hpairs) as (Ha & _).
        destruct (Ha v v' kind Hin') as (p & _ & Hp').
        exact (leaves_apart M M_distinct b x fb fields v' Hb Hfields
                 (not_eq_sym Hxb) (upath_leaf_in fb p v' kind Hp') Hv').
    + intros v kind s Hin.
      destruct (forall2_concat_in _ _ _ _ Hl Hin)
        as ([f u] & l0 & _ & Hent & Hin0).
      destruct u as [v0 | nested]; cbn [alloc_entry] in Hent.
      * destruct Hent as (kind0 & s0 & -> & Hf & Hplain).
        destruct Hin0 as [Heq | []]. injection Heq as <- <- <-.
        destruct (alloc_field_eval statics vars d w f kind0 s0 Hw Hf)
          as (w' & Hs & _ & Hkw).
        exists w'. split; [ | exact Hkw ].
        rewrite <- (plain_simple_eval M vars vars' s0 Hvars Hplain). exact Hs.
      * destruct Hent as (b & fb & pairs & Hf & Hb & Hxb & Hpairs & ->).
        destruct (alloc_field_eval statics vars d w f Value (Var b) Hw Hf)
          as (wb & Hs & _ & _).
        simpl in Hs.
        destruct (pairs_hold vars' nested fb wb pairs Hpairs
                    (proj2 Hvars b fb wb Hb Hs))
          as (_ & Hseconds & _).
        apply in_pairs_lets in Hin0 as (v' & -> & Hin').
        destruct (Hseconds v v' kind Hin') as [_ (w' & Hl' & Hk')].
        exists w'. split; [ exact Hl' | exact Hk' ].
    + intros p v k Hpath.
      apply upath_unboxed_inv in Hpath as (f & u & p0 & -> & Hfu & Hp0).
      destruct (forall2_in_both _ _ _ _ Hl Hfu) as (l0 & Hl0 & Hent).
      destruct u as [v0 | nested]; cbn [alloc_entry] in Hent.
      * apply upath_not_unboxed_inv in Hp0 as (-> & -> & ->).
        destruct Hent as (kind0 & s0 & -> & Hf & Hplain).
        pose proof (alloc_field_kind d f kind0 s0 Hf) as Hk0. subst kind0.
        destruct (alloc_field_eval statics vars d w f (field_kind f) s0 Hw Hf)
          as (w' & Hs & Hfw & _).
        exists s0, w'. split; [ exact (in_concat_of _ _ _ Hl0 (or_introl
                                                       eq_refl)) | ].
        split; [ rewrite <- (plain_simple_eval M vars vars' s0 Hvars Hplain);
                 exact Hs | ].
        cbn [path_value]. rewrite Hfw. reflexivity.
      * destruct Hent as (b & fb & pairs & Hf & Hb & Hxb & Hpairs & ->).
        destruct (alloc_field_eval statics vars d w f Value (Var b) Hw Hf)
          as (wb & Hs & Hfw & _).
        simpl in Hs.
        destruct (pairs_hold vars' nested fb wb pairs Hpairs
                    (proj2 Hvars b fb wb Hb Hs))
          as (_ & _ & Hpaths).
        destruct (Hpaths p0 v k (upath_unboxed_kind _ Value _ _ _ _ Hp0))
          as (v' & w' & Hin' & Hl' & Hpw).
        exists (Var v'), w'.
        split; [ exact (in_concat_of _ _ _ Hl0
                          (pairs_in_lets pairs v v' k Hin')) | ].
        split; [ exact Hl' | ]. cbn [path_value]. rewrite Hfw. exact Hpw.
    + apply (run_answer (relatedh M) _ _ _ _ (length (concat ls)) Hrun).
      * exists K. unfold relatedh.
        cbn [state_expr state_vars state_conts state_statics state_heap].
        split; [ exact Hbody | split; [ exact Hrel | ] ].
        split; [ exact Hconts | repeat split ].
      * intros _. unfold measure. simpl. lia.
  - (* An alias of an unboxed variable. *)
    rewrite step_let_simple. cbn [eval_simple].
    destruct (lookup y vars) as [w | ] eqn:Hw; [ | exact I ].
    destruct (kind_eqb (kind_of_value w) Value) eqn:Hk; [ | exact I ].
    destruct (pairs_hold vars' fields fields_y w pairs Hl
                (proj2 Hvars y fields_y w Hy Hw))
      as (Hfirsts & Hseconds & Hpaths).
    destruct (lets_case vars vars' x fields w (pairs_lets pairs) body' conts'
                statics hp rcd Hvars Hfields)
      as (vars1 & Hrun & Hrel).
    + rewrite pairs_lets_variables. exact Hfirsts.
    + intros v kind s v' Hin Hv' ->.
      apply in_pairs_lets in Hin as (v'' & Heq & Hin). injection Heq as <-.
      destruct (Hseconds v v' kind Hin) as [Hleaf _].
      exact (leaves_apart M M_distinct y x fields_y fields v' Hy Hfields
               (not_eq_sym Hne) Hleaf Hv').
    + intros v kind s Hin.
      apply in_pairs_lets in Hin as (v' & -> & Hin).
      destruct (Hseconds v v' kind Hin) as [_ (w' & Hl' & Hk')].
      exists w'. split; [ exact Hl' | exact Hk' ].
    + intros p v k Hpath.
      destruct (Hpaths p v k Hpath) as (v' & w' & Hin & Hl' & Hpw).
      exists (Var v'), w'.
      split; [ exact (pairs_in_lets pairs v v' k Hin) | ].
      split; [ exact Hl' | exact Hpw ].
    + apply (run_answer (relatedh M) _ _ _ _ (length (pairs_lets pairs)) Hrun).
      * exists K. unfold relatedh.
        cbn [state_expr state_vars state_conts state_statics state_heap].
        split; [ exact Hbody | split; [ exact Hrel | ] ].
        split; [ exact Hconts | repeat split ].
      * intros _. unfold measure. simpl. lia.
  - (* The load of a field unboxed in turn. *)
    rewrite step_let_exec.
    destruct (exec_named statics hp rcd vars d) as [[[r hp1] rcd1] | ]
      eqn:Hxr; [ | exact I ].
    destruct (read_exec statics hp rcd vars d f s r hp1 rcd1 Hread Hxr)
      as (-> & -> & w0 & Hw0 & Hnc & Hfield).
    destruct (kind_eqb (kind_of_value r) Value) eqn:Hk; [ | exact I ].
    destruct Hs as [-> | (k & ->)];
      [ | simpl in Hw0; injection Hw0 as <-; exfalso; exact (Hnc _ eq_refl) ].
    simpl in Hw0.
    pose proof (fields_hold_nested vars' fields_y nested f w0 r
                  (proj2 Hvars y fields_y w0 Hy Hw0) Hin Hfield) as Hhold.
    destruct (pairs_hold vars' fields nested r pairs Hl Hhold)
      as (Hfirsts & Hseconds & Hpaths).
    destruct (lets_case vars vars' x fields r (pairs_lets pairs) body' conts'
                statics hp rcd Hvars Hfields)
      as (vars1 & Hrun & Hrel).
    + rewrite pairs_lets_variables. exact Hfirsts.
    + intros v kind0 s v' Hin0 Hv' ->.
      apply in_pairs_lets in Hin0 as (v'' & Heq & Hin0).
      injection Heq as <-.
      destruct (Hseconds v v' kind0 Hin0) as [Hleaf _].
      exact (leaves_apart M M_distinct y x fields_y fields v' Hy Hfields
               (not_eq_sym Hne)
               (leaves_unboxed_in Value fields_y f (Unboxed nested) v' Hin
                  Hleaf)
               Hv').
    + intros v kind0 s Hin0.
      apply in_pairs_lets in Hin0 as (v' & -> & Hin0).
      destruct (Hseconds v v' kind0 Hin0) as [_ (w' & Hl' & Hk')].
      exists w'. split; [ exact Hl' | exact Hk' ].
    + intros p v k0 Hpath.
      destruct (Hpaths p v k0 Hpath) as (v' & w' & Hin0 & Hl' & Hpw).
      exists (Var v'), w'.
      split; [ exact (pairs_in_lets pairs v v' k0 Hin0) | ].
      split; [ exact Hl' | exact Hpw ].
    + apply (run_answer (relatedh M) _ _ _ _ (length (pairs_lets pairs)) Hrun).
      * exists K. unfold relatedh.
        cbn [state_expr state_vars state_conts state_statics state_heap].
        split; [ exact Hbody | split; [ exact Hrel | ] ].
        split; [ exact Hconts | repeat split ].
      * intros _. unfold measure. simpl. lia.
  - (* A read of a field. *)
    rewrite step_let_exec.
    destruct (exec_named statics hp rcd vars d) as [[[r hp1] rcd1] | ]
      eqn:Hxr; [ | exact I ].
    destruct (read_exec statics hp rcd vars d f s r hp1 rcd1 Hread Hxr)
      as (-> & -> & w0 & Hw0 & Hnc & Hfield).
    destruct (kind_eqb (kind_of_value r) kind) eqn:Hk; [ | exact I ].
    destruct Hs as [-> | (k & ->)];
      [ | simpl in Hw0; injection Hw0 as <-; exfalso; exact (Hnc _ eq_refl) ].
    simpl in Hw0.
    assert (Hp : upath Value (Unboxed fields) [f] v (field_kind f))
      by (econstructor; [ exact Hin | constructor ]).
    destruct (proj2 Hvars y fields w0 Hfields Hw0 [f] v _ Hp)
      as (w' & Hpw & Hv & _).
    cbn [path_value] in Hpw.
    destruct (field_value f w0) as [w1 | ] eqn:Hfw; [ | discriminate Hpw ].
    injection Hpw as <-.
    rewrite <- (Hfield w1 eq_refl) in Hv.
    right. left. eexists. split.
    + rewrite step_let_simple. cbn [eval_simple]. rewrite Hv, Hk.
      reflexivity.
    + exists K. split; [ exact Hbody | ].
      split; [ exact (env_rel_bind M _ _ x r Hvars Hx) | ].
      split; [ exact Hconts | repeat split ].
  - (* A [Let_static]. *)
    rewrite step_let_static,
      (local_bindings_eval M vars vars' bindings Hvars Hbindings statics).
    destruct (eval_static_bindings vars' bindings statics) as [statics1 | ]
      eqn:Hst; [ | exact I ].
    right. left. eexists.
    split; [ rewrite step_let_static, Hst; reflexivity | ].
    exists K. split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ exact Hconts | repeat split ].
  - (* A non-recursive [Let_cont]. *)
    rewrite step_let_cont_non_recursive. right. left. eexists.
    split; [ rewrite step_let_cont_non_recursive; reflexivity | ].
    exists ((k, sh) :: K). split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ | repeat split ].
    constructor; [ | exact Hconts ].
    exact (cont_relh_closure M sh ps ps' md md' h h' vars vars' K [] conts
             conts' [] [] Hps Hmd Hmd' Hh Hvars Hconts
             (ubh_handlers_nil M K)).
  - (* A recursive [Let_cont]. *)
    rewrite step_let_cont_recursive.
    destruct (handlers_bind_statics handlers) eqn:Hb; [ exact I | ].
    right. left. eexists.
    split.
    { rewrite step_let_cont_recursive.
      destruct (handlers_bind_statics handlers') eqn:Hb';
        [ congruence | reflexivity ]. }
    exists (S ++ K). split; [ exact Hbody | split; [ exact Hvars | ] ].
    split; [ apply rec_closures_relh; assumption | repeat split ].
  - (* An [Apply_cont]. *)
    destruct Hac as (Hk & Htrap & Hargs).
    rewrite step_apply_cont.
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
      [ | exact I ].
    unfold apply_continuation.
    destruct (resolve_relh M K conts conts' (ac_continuation ac) Hconts)
      as [(HK & Hc & Hc') | (sh & HK & Hc)].
    + unfold shapes_at in Hargs. rewrite HK in Hargs.
      destruct (args_rel_nil_inv _ _ Hargs) as [Heq Hplain].
      rewrite Hc. cbn [apply_cont_value]. exists vs, rcd.
      split; [ | reflexivity ].
      rewrite step_apply_cont, Heq,
        <- (plain_simples_eval M vars vars' _ Hvars Hplain), Hvs.
      unfold apply_continuation. rewrite Hk, Hc'. reflexivity.
    + unfold shapes_at in Hargs. rewrite HK in Hargs.
      remember (resolve (ac_continuation ac) conts) as c eqn:Ec.
      remember (resolve (ac_continuation ac) conts') as c' eqn:Ec'.
      destruct Hc as [sh ps ps' md md' h h' cvars cvars' Kc S cconts cconts'
                        rec rec' Hps Hmd Hmd' Hh Hcv Hcc Hrec].
      cbn [apply_cont_value]. rewrite Hmd.
      destruct (kinds_eqb (map kind_of_value vs) (map snd ps)) eqn:Hkinds;
        [ | exact I ].
      apply FlambdaSemanticsProofs.kinds_eqb_eq in Hkinds.
      destruct (bind_params (map fst ps) vs cvars) as [hv | ] eqn:Hbind;
        [ | exact I ].
      destruct (bind_args_rel sh _ _ Hargs ps ps' vars vars' cvars cvars' hv vs
                  Hps Hvars Hcv Hvs Hkinds Hbind)
        as (vs' & hv' & Hvs' & Hk' & Hb' & Hrel).
      right. left. eexists. split.
      * rewrite step_apply_cont, Hvs'. unfold apply_continuation.
        rewrite Hk, <- Ec'. cbn [apply_cont_value]. rewrite Hmd', <- Hk'.
        rewrite (proj2 (FlambdaSemanticsProofs.kinds_eqb_eq _ _) eq_refl), Hb'.
        reflexivity.
      * exists (S ++ Kc). split; [ exact Hh | split; [ exact Hrel | ] ].
        split; [ apply rec_closures_relh; assumption | repeat split ].
  - (* A [Switch]. *)
    rewrite step_switch,
      (plain_simple_eval M vars vars' scrutinee Hvars Hscrutinee).
    destruct (eval_simple vars' scrutinee) as [w | ] eqn:Hw; [ | exact I ].
    destruct (naked_immediate_of_value w) as [i | ] eqn:Hi; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
    destruct (find_arm_rel K i arms arms' Harms ac Harm) as (ac' & Harm' & Hac).
    right. left. eexists.
    split; [ rewrite step_switch, Hw, Hi, Harm'; reflexivity | ].
    exists K. split; [ constructor; exact Hac | ].
    split; [ exact Hvars | split; [ exact Hconts | repeat split ] ].
  - rewrite step_invalid. exact I.
Qed.

(* Related terms, run on their own, refine each other. *)
Lemma ubh_refines (p q : expr) :
  ubh M [] p q ->
  refines (State p [] [] empty_static_env empty_heap no_reads)
    (State q [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hub. apply (shrinking_refines (relatedh M) eq measure).
  - exact relatedh_answered.
  - intros vs vs' ->. reflexivity.
  - exists []. split; [ exact Hub | split; [ apply env_rel_empty | ] ].
    split; [ constructor | repeat split ].
Qed.


End Simulation_heap.

(* ======================================================================== *)
(* The two rebuilds of a unit of the fragment, with the heap                *)
(* ======================================================================== *)

Section Rebuilds_heap.


Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable preserve_direct_calls : bool.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable first_fresh first_fresh_u : positive.


(* What the graph has of the unit: the facts of FlambdaReaperFlowProofs.v
   ([expr_edges]), and a solution closed under the rules. *)
Variable noted : symbol -> Prop.
Variable deps : closure_dep -> Prop.
Variable cds : finite_map code_dep.
Variable kept_slots : value_slot -> bool.
Variable codes0 : list (code_id * code).
Variable symbols : list symbol.
Variable is_local : field -> bool.
Hypothesis sol_closed : closed is_local (graph_of_facts F) sol.
Hypothesis local_slots : is_local_slots_only is_local.
(* Each unboxed field is read, from a use of the variable, into a node
   that has a use ([get_fields]). *)
Hypothesis fields_read : ∀ (x : variable) (fields : unboxed_fields)
    (f : field) (u : unboxed_field),
  fields_of unboxed x = Some fields -> (f, u) ∈ fields ->
  ∃ w base : node,
    g_accessor (graph_of_facts F) w f base
    ∧ s_usages sol (Node_variable x) base = true
    ∧ s_has_usage sol w = true.
(* A read that the rebuild keeps, into a variable that is not unboxed,
   of an unboxed variable, finds its field among the unboxed ones, a
   variable ([check_unboxed_reads]). *)
Hypothesis reads_found : ∀ (x y : variable) (kind : flambda_kind)
    (f : field) (fields : unboxed_fields),
  fields_of unboxed x = None -> fields_of unboxed y = Some fields ->
  g_accessor (graph_of_facts F) (Node_variable x) f (Node_variable y) ->
  is_dead_var sol x kind = false -> is_var_used sol x kind = true ->
  kind = Value ∨ (∃ nk : naked_number_kind, kind = Naked_number nk) ->
  ∃ v : variable, Keyed.lookup field_eqb f fields = Some (Not_unboxed v).


Abbreviation edges ce e :=
  (FlambdaReaperFlowProofs.expr_edges (graph_of_facts F) noted deps cds
     kept_slots codes0 symbols True True None ce e).

Abbreviation kept_simple :=
  (URP.kept_simple sol).
Abbreviation field_kept :=
  (URP.field_kept F sol unboxed is_local sol_closed local_slots fields_read).
Abbreviation plain_rebuild :=
  (URP.plain_rebuild F sol preserve_direct_calls unit_code_ids first_fresh).
Abbreviation unbox_rebuild :=
  (URP.unbox_rebuild F sol cannot_change code_deps preserve_direct_calls
     unit_code_ids codes unboxed first_fresh_u).
Abbreviation shape_of_param :=
  (URP.shape_of_param unboxed).
Abbreviation K_of :=
  (URP.K_of unboxed).
Abbreviation decision_match :=
  (URP.decision_match unboxed).
Abbreviation conts_match :=
  (URP.conts_match unboxed).
Abbreviation plain_not_bound :=
  (URP.plain_not_bound unboxed).
Abbreviation rewrite_simple_u :=
  (URP.rewrite_simple_u sol unboxed).
Abbreviation rewrite_simple_plain :=
  (URP.rewrite_simple_plain sol unboxed).
Abbreviation rewrite_simples_plain :=
  (URP.rewrite_simples_plain sol unboxed).
Abbreviation rebuild_fields_u :=
  (URP.rebuild_fields_u sol unboxed).
Abbreviation rebuild_fields_plain :=
  (URP.rebuild_fields_plain sol unboxed).
Abbreviation rebuild_fields_nth :=
  (URP.rebuild_fields_nth sol).
Abbreviation named_plain_u :=
  (URP.named_plain_u sol unboxed).
Abbreviation named_plain_rebuilt :=
  (URP.named_plain_rebuilt sol unboxed).
Abbreviation kinds_env_kept :=
  (URP.kinds_env_kept sol).
Abbreviation kept_rewrite :=
  (URP.kept_rewrite sol).
Abbreviation bind_boxed_number_lets :=
  (URP.bind_boxed_number_lets unboxed).
Abbreviation params_to_keep_length :=
  (URP.params_to_keep_length sol).
Abbreviation get_args_decided :=
  (URP.get_args_decided sol unboxed).
Abbreviation unboxed_binding_cases :=
  (URP.unboxed_binding_cases sol unboxed).
Abbreviation plain_binding_cases :=
  (URP.plain_binding_cases sol unboxed).
Abbreviation rewrite_simple_kinds :=
  (URP.rewrite_simple_kinds sol).
Abbreviation rebuild_fields_kinds :=
  (URP.rebuild_fields_kinds sol).
Abbreviation rebuild_named_kinds :=
  (URP.rebuild_named_kinds sol).
Abbreviation conts_match_lookup :=
  (URP.conts_match_lookup unboxed).
Abbreviation unboxed_args_pairs :=
  (URP.unboxed_args_pairs unboxed).
Abbreviation get_args_rel :=
  (URP.get_args_rel sol unboxed).
Abbreviation rewrite_apply_cont_rel :=
  (URP.rewrite_apply_cont_rel sol unboxed).
Abbreviation params_rel_handler :=
  (URP.params_rel_handler unboxed).
Abbreviation handler_match :=
  (URP.handler_match sol unboxed).
Abbreviation conts_match_rec :=
  (URP.conts_match_rec sol unboxed).
Abbreviation bind_block_fields_lets2 :=
  (URP.bind_block_fields_lets2 sol unboxed).
Abbreviation ac_rel_conts :=
  (URP.ac_rel_conts unboxed).
Abbreviation rebuild_let_u_plain :=
  (URP.rebuild_let_u_plain sol unboxed).
Abbreviation plain_rebuild_let_static :=
  (URP.plain_rebuild_let_static F sol preserve_direct_calls unit_code_ids
     first_fresh).
Abbreviation unbox_rebuild_let_static :=
  (URP.unbox_rebuild_let_static F sol cannot_change code_deps
     preserve_direct_calls unit_code_ids codes unboxed first_fresh_u).
Abbreviation plain_rebuild_let_cont :=
  (URP.plain_rebuild_let_cont F sol preserve_direct_calls unit_code_ids
     first_fresh).
Abbreviation unbox_rebuild_let_cont :=
  (URP.unbox_rebuild_let_cont F sol cannot_change code_deps
     preserve_direct_calls unit_code_ids codes unboxed first_fresh_u).
Abbreviation plain_rebuild_let_cont_rec :=
  (URP.plain_rebuild_let_cont_rec F sol preserve_direct_calls unit_code_ids
     first_fresh).
Abbreviation unbox_rebuild_let_cont_rec :=
  (URP.unbox_rebuild_let_cont_rec F sol cannot_change code_deps
     preserve_direct_calls unit_code_ids codes unboxed first_fresh_u).
Abbreviation plain_rebuild_handler :=
  (URP.plain_rebuild_handler F sol preserve_direct_calls unit_code_ids
     first_fresh).
Abbreviation unbox_rebuild_handler :=
  (URP.unbox_rebuild_handler F sol cannot_change code_deps
     preserve_direct_calls unit_code_ids codes unboxed first_fresh_u).
Abbreviation local_bindings_rebuilt :=
  (URP.local_bindings_rebuilt F sol cannot_change code_deps
     preserve_direct_calls unit_code_ids codes unboxed first_fresh
     first_fresh_u).
Definition ubh_result (K : finite_map (list (option unboxed_fields)))
    (p q : expr) : Prop :=
  ubh unboxed K p q
  ∧ free_continuations_of_expr p = free_continuations_of_expr q
  ∧ binds_statics p = binds_statics q.

Lemma ubh_result_invalid (K : finite_map (list (option unboxed_fields))) :
  ubh_result K Invalid Invalid.
Proof. split; [ constructor | split; reflexivity ]. Qed.


Lemma rebuild_ubh :
  (∀ e : expr,
     ∀ (en : unbox_env) (C : finite_map (list (variable * bool)))
       (ce : finite_map (list node)),
       uinv en -> conts_match C (ue_decisions en) ->
       local_expr unboxed sol (env_kinds (ue_env en)) C e = true ->
       edges ce e ->
       ubh_result (K_of C) (plain_rebuild (pen_of en C) e)
         (unbox_rebuild en e))
  ∧ (∀ h : continuation_handler,
       ∀ (en : unbox_env) (C : finite_map (list (variable * bool)))
         (ce : finite_map (list node)) (is_exn : bool),
         uinv en -> conts_match C (ue_decisions en) ->
         local_handler unboxed sol (env_kinds (ue_env en)) C
           (handler_mask sol is_exn h) h
         = true ->
         FlambdaReaperFlowProofs.handler_edges (graph_of_facts F) noted deps
           cds kept_slots codes0 symbols True True None ce h ->
         ∀ dead : bool,
           ∃ (ps ps' : list (variable * flambda_kind))
             (metadata metadata' : handler_metadata) (hp hu : expr),
             rebuild_handler F sol preserve_direct_calls unit_code_ids
               first_fresh (pen_of en C) dead (handler_mask sol is_exn h) h
             = Handler (map fst ps) metadata hp
             ∧ rebuild_handler_u false F sol cannot_change code_deps
                 preserve_direct_calls unit_code_ids codes unboxed
                 first_fresh_u [] en dead
                 (handler_decisions false sol unboxed is_exn h) h
               = Handler (map fst ps') metadata' hu
             ∧ params_rel unboxed
                 (flat_map shape_of_param
                    (combine (handler_variables h)
                       (handler_mask sol is_exn h)))
                 ps ps'
             ∧ hm_params_kinds metadata = map snd ps
             ∧ hm_params_kinds metadata' = map snd ps'
             ∧ Forall2 decision_match
                 (combine (handler_variables h) (handler_mask sol is_exn h))
                 (handler_decisions false sol unboxed is_exn h)
             ∧ length (handler_mask sol is_exn h)
               = length (handler_variables h)
             ∧ ubh_result (K_of C) hp hu).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - (* Let *)
    intros x kind d body IH en C ce Hen HC Hloc Hedges. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hloc Hbody].
    apply andb_true_iff in Hloc as [Hxf Hbind]. apply negb_true_iff in Hxf.
    destruct Hedges as [Hnamed_edges Hbody_edges].
    specialize (IH (ubind_variables en [(x, kind)]) C ce
                  (uinv_bind en [(x, kind)]) HC Hbody Hbody_edges).
    rewrite pen_bind in IH.
    unfold plain_rebuild, unbox_rebuild in *. cbn [rebuild_expr rebuild_expr_u].
    unfold is_pending_projection.
    rewrite (uinv_projections en Hen), pen_projections.
    rewrite (rebuild_named_kinds (pen_of en C) (ue_env en) x d eq_refl).
    assert (Hd : (∃ s, d = Simple s) ∨ (∃ p args, d = Prim p args)).
    { destruct (fields_of unboxed x) as [fields | ] eqn:Hx.
      - destruct (unboxed_binding_cases _ x kind d fields Hbind)
          as (_ & [(tag & args & -> & _) | [(bn & c & -> & _)
                  | [(y & fy & -> & _)
                    | (bk & mut & i & y & fy & n & -> & _)]]]);
          [ right | right | left | right ]; eauto.
      - destruct (plain_binding_cases x kind d Hbind)
          as [Hp | (p & y & f & fs & -> & _)]; [ | right; eauto ].
        destruct d; try discriminate Hp; eauto. }
    destruct IH as (IHub & IHfc & IHbs).
    destruct Hd as [(s0 & ->) | (p & args & ->)]; cbn beta iota;
      (destruct (is_dead_var sol x kind) eqn:Hdead;
       [ exact (ubh_result_invalid _) | ]);
      (destruct (is_var_used sol x kind) eqn:Hused;
       [ | split; [ exact IHub | split; assumption ] ]);
      change (unboxed_of_node unboxed (Node_variable x))
        with (fields_of unboxed x);
      change (changed_of_node [] (Node_variable x)) with (@None closure_repr);
      cbn iota;
      (destruct (fields_of unboxed x) as [fields | ] eqn:Hx).
    + (* An alias of an unboxed variable. *)
      destruct (unboxed_binding_cases _ x kind _ fields Hbind)
        as (-> & [(tag & args & Heq & _) | [(bn & c & Heq & _)
                 | [(y & fields_y & Heq & Hne & Hy & Hflat & Hyused)
                   | (bk & mut & i & y & fy & n & Heq & _)]]]);
        try discriminate Heq.
      injection Heq as ->.
      cbn [rebuild_named rewrite_simple].
      unfold kind_of_variable in Hyused |- *.
      cbn [env_kinds kinds_env] in Hyused. rewrite Hyused.
      unfold rebuild_unboxed_binding.
      destruct (bind_fields_pairs fields fields_y
                  (rebuild_expr_u false F sol cannot_change code_deps
                     preserve_direct_calls unit_code_ids codes unboxed
                     first_fresh_u [] (ubind_variables en [(x, Value)]) body)
                  Hflat)
        as (pairs & Hentries & Hl).
      cbn [unboxed_of].
      change (unboxed_of_node unboxed (Node_variable y))
        with (fields_of unboxed y).
      rewrite Hy, Hl.
      destruct (lets_free_continuations (pairs_lets pairs)
                  (rebuild_expr_u false F sol cannot_change code_deps
                     preserve_direct_calls unit_code_ids codes unboxed
                     first_fresh_u [] (ubind_variables en [(x, Value)]) body))
        as [Hfc Hbs].
      split; [ exact (ubh_alias unboxed _ x y fields fields_y pairs _ _ Hx Hxf
                        Hy Hne Hentries IHub)
             | ].
      cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
      split; assumption.
    + (* A simple. *)
      destruct (plain_binding_cases x kind _ Hbind)
        as [Hp | (p & y & f & fs & Heq & _)]; [ | discriminate Heq ].
      rewrite (rebuild_let_u_plain _ x kind _ _ Hp).
      split; [ | split; assumption ].
      constructor; [ | apply named_plain_rebuilt, Hp | exact IHub ].
      apply plain_variable_iff. unfold Keyed.is_bound.
      unfold fields_of in Hx. rewrite Hx.
      split; [ reflexivity | exact Hxf ].
    + (* An unboxed allocation, or the load of a field unboxed in turn. *)
      destruct (unboxed_binding_cases _ x kind _ fields Hbind)
        as (-> & [(tag & args' & Heq & Hfields)
                 | [(bn & c & Heq & Hc & Hfields)
                   | [(y & fy & Heq & _)
                     | (bk & mut & i & y & fy & nested & Heq & Hne & Hy & Hny
                        & Halike)]]]);
        [ | | discriminate Heq | ];
        injection Heq as -> ->; simpl in Hnamed_edges;
        destruct Hnamed_edges as [_ Hnamed_edges].
      * destruct (bind_block_fields_lets2 (ue_env en) x tag args' fields)
          with (body := rebuild_expr_u false F sol cannot_change code_deps
                          preserve_direct_calls unit_code_ids codes unboxed
                          first_fresh_u [] (ubind_variables en [(x, Value)])
                          body)
          as (ls & Hl & Hentries).
        { intros f u Hin.
          destruct (Hfields f u Hin) as (i & a & -> & Hnth & Hkind & Hcase).
          rewrite kinds_env_kind in Hkind.
          destruct (field_kept x fields _ u a (ue_env en) Hx Hin
                      (proj1 Hnamed_edges i a Hnth) eq_refl)
            as [Hused' Hkept].
          exists i, a. repeat split; assumption. }
        unfold rebuild_unboxed_binding. rewrite Hl. cbn [rebuild_named].
        destruct (lets_free_continuations (concat ls)
                    (rebuild_expr_u false F sol cannot_change code_deps
                       preserve_direct_calls unit_code_ids codes unboxed
                       first_fresh_u [] (ubind_variables en [(x, Value)])
                       body))
          as [Hfc Hbs].
        split; [ exact (ubh_alloc unboxed _ x _ fields ls _ _ Hx Hxf Hentries
                          eq_refl IHub)
               | ].
        cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
        split; assumption.
      * destruct (bind_boxed_number_lets x bn c fields
                    (rebuild_named sol (ue_env en) x
                       (Prim (Unary (Box_number bn)) [c])) Hc)
          with (body := rebuild_expr_u false F sol cannot_change code_deps
                          preserve_direct_calls unit_code_ids codes unboxed
                          first_fresh_u [] (ubind_variables en [(x, Value)])
                          body)
          as (ls & Hl & Hentries).
        { intros Hne. destruct fields as [ | [f u] fields]; [ easy | ].
          destruct (Hfields f u (or_introl eq_refl)) as (v & -> & ->).
          destruct (field_kept x _ _ (Not_unboxed v) c (ue_env en) Hx
                      (or_introl eq_refl) (proj1 Hnamed_edges) eq_refl)
            as [Hused' Hkept].
          cbn [rebuild_named].
          unfold field_used. rewrite Hused'. simpl.
          rewrite (kept_rewrite _ c Hkept). reflexivity. }
        { intros f u Hin. exact (Hfields f u Hin). }
        unfold rebuild_unboxed_binding. rewrite Hl.
        destruct (lets_free_continuations (concat ls)
                    (rebuild_expr_u false F sol cannot_change code_deps
                       preserve_direct_calls unit_code_ids codes unboxed
                       first_fresh_u [] (ubind_variables en [(x, Value)])
                       body))
          as [Hfc Hbs].
        split; [ refine (ubh_alloc unboxed _ x _ fields _ _ _ Hx Hxf Hentries
                           _ IHub);
                 cbn [rebuild_named]; destruct (field_used _ _ _);
                 reflexivity
               | ].
        cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
        split; assumption.
      * (* The load of a field unboxed in turn. *)
        cbn [rebuild_named]. unfold rebuild_unboxed_binding, load_unboxed_field.
        cbn [unboxed_of].
        change (unboxed_of_node unboxed (Node_variable y))
          with (fields_of unboxed y).
        rewrite Hy, Hny.
        destruct (bind_fields_pairs fields nested
                    (rebuild_expr_u false F sol cannot_change code_deps
                       preserve_direct_calls unit_code_ids codes unboxed
                       first_fresh_u [] (ubind_variables en [(x, Value)])
                       body)
                    Halike)
          as (pairs & Hentries & Hl).
        rewrite Hl.
        destruct (lets_free_continuations (pairs_lets pairs)
                    (rebuild_expr_u false F sol cannot_change code_deps
                       preserve_direct_calls unit_code_ids codes unboxed
                       first_fresh_u [] (ubind_variables en [(x, Value)])
                       body))
          as [Hfc Hbs].
        split.
        -- apply (ubh_load unboxed _ x Value _ (Field_block (Z.to_nat i) Value)
                    (rewrite_simple sol (ue_env en) (Var y)) y fields fy
                    nested pairs); try assumption.
           ++ reflexivity.
           ++ simpl. destruct (is_var_used sol y _); [ left; reflexivity | ].
              right. eexists. reflexivity.
           ++ exact (keyed_lookup_in field_eqb field_eqb_decides _ _ fy Hny).
        -- cbn [free_continuations_of_expr binds_statics]. rewrite Hfc, Hbs.
           split; assumption.
    + (* A primitive of a variable that is not unboxed. *)
      destruct (plain_binding_cases x kind _ Hbind)
        as [Hp | (p' & y & f & fs & Heq & Hread & Hreads & Hnamed & Hy
                  & Hfield)].
      * rewrite (rebuild_let_u_plain _ x kind _ _ Hp).
        split; [ | split; assumption ].
        constructor; [ | apply named_plain_rebuilt, Hp | exact IHub ].
        apply plain_variable_iff. unfold Keyed.is_bound.
        unfold fields_of in Hx. rewrite Hx.
        split; [ reflexivity | exact Hxf ].
      * rewrite Heq in Hnamed, Hnamed_edges |- *.
        rewrite (Hnamed (ue_env en)).
        assert (Hacc : g_accessor (graph_of_facts F) (Node_variable x) f
                         (Node_variable y)).
        { clear -Hread Hnamed_edges.
          destruct p'; try discriminate Hread; simpl in Hread;
            injection Hread as <-; exact (proj1 (proj2 Hnamed_edges)). }
        destruct (reads_found x y kind f fs Hx Hy Hacc Hdead Hused Hfield)
          as (v & Hv).
        unfold rebuild_let_u.
        assert (Hfr : field_read (Prim (Unary p') [Var y]) = Some (f, Var y)).
        { clear -Hread. destruct p'; try discriminate Hread; exact Hread. }
        rewrite Hfr. cbn [unboxed_of]. change (unboxed_of_node unboxed
          (Node_variable y)) with (fields_of unboxed y).
        rewrite Hy, Hv.
        split; [ | split; assumption ].
        apply (ubh_read unboxed _ x kind
                 (Prim (Unary p') [rewrite_simple sol (ue_env en) (Var y)]) f
                 (rewrite_simple sol (ue_env en) (Var y)) y v fs).
        -- apply plain_variable_iff. unfold Keyed.is_bound.
           unfold fields_of in Hx. rewrite Hx.
           split; [ reflexivity | exact Hxf ].
        -- apply Hreads.
        -- simpl. destruct (is_var_used sol y _); [ left; reflexivity | ].
           right. eexists. reflexivity.
        -- exact Hy.
        -- exact (keyed_lookup_in field_eqb field_eqb_decides f _ fs Hv).
        -- exact IHub.
  - (* Let_static *)
    intros bindings body _ IH en C ce Hen HC Hloc Hedges. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hbs Hbody].
    destruct (IH (ubind_variables en []) C ce (uinv_bind en []) HC Hbody
                (proj2 Hedges))
      as (IHub & IHfc & IHbs).
    rewrite pen_bind in IHub, IHfc, IHbs.
    destruct (local_bindings_rebuilt en C bindings Hbs) as [Heq Hplain].
    rewrite plain_rebuild_let_static, unbox_rebuild_let_static, Heq.
    split; [ constructor; [ exact Hplain | exact IHub ] | ].
    cbn [free_continuations_of_expr binds_statics]. rewrite IHfc, IHbs.
    split; reflexivity.
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH en C ce Hen HC Hloc Hedges. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hh Hbody].
    destruct Hedges as [Hh_edges Hbody_edges].
    rewrite plain_rebuild_let_cont, unbox_rebuild_let_cont. cbv zeta.
    destruct (IHh en C ce (handler_is_exn_handler h) Hen HC Hh Hh_edges true)
      as (ps & ps' & md & md' & hp & hu & Hp & Hu & Hrel & Hmd & Hmd'
          & Hmatch & Hlen & Hhub & Hhfc & Hhbs).
    set (is_exn := handler_is_exn_handler h) in *.
    set (mask := handler_mask sol is_exn h) in *.
    set (ds := handler_decisions false sol unboxed is_exn h) in *.
    set (C' := (k, combine (handler_variables h) mask) :: C).
    assert (HC' : conts_match C' (ue_decisions
                                   (ubind_continuations en [(k, ds)]))).
    { constructor; [ split; [ reflexivity | exact Hmatch ] | exact HC ]. }
    assert (Hpen : pen_of (ubind_continuations en [(k, ds)]) C'
                   = bind_continuations (pen_of en C) [(k, mask)]).
    { unfold pen_of, C'. cbn [map env_conts env_kinds ue_env
        ubind_continuations bind_continuations app].
      rewrite map_snd_combine by exact (eq_sym Hlen). reflexivity. }
    destruct (IH (ubind_continuations en [(k, ds)]) C' _
                (uinv_continuations en [(k, ds)]) HC' Hbody Hbody_edges)
      as (IHub & IHfc & IHbs).
    rewrite Hpen in IHub, IHfc, IHbs.
    rewrite <- IHfc.
    destruct (among k _) eqn:Hk.
    + rewrite Hp, Hu.
      split; [ eapply ubh_let_cont; eassumption | ].
      cbn [free_continuations_of_expr free_continuations_of_handler
           binds_statics handler_binds_statics].
      rewrite Hhfc, IHfc, Hhbs, IHbs. split; reflexivity.
    + split; [ | split; assumption ].
      apply (ubh_agree unboxed _ _ _ IHub). intros k' Hk'.
      unfold C', K_of. cbn [map]. rewrite lookup_cons.
      destruct (Pos.eqb k' k) eqn:E; [ | reflexivity ].
      apply Pos.eqb_eq in E. subst k'. apply among_In in Hk'.
      rewrite Hk' in Hk. discriminate.
  - (* Let_cont_recursive *)
    intros handlers body IHh IH en C ce Hen HC Hloc Hedges. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hhs Hbody].
    rewrite forallb_forall in Hhs.
    destruct Hedges as [Hhs_edges Hbody_edges].
    rewrite plain_rebuild_let_cont_rec, unbox_rebuild_let_cont_rec. cbv zeta.
    set (dss := map (fun '(k, h) =>
                       (k, handler_decisions false sol unboxed false h))
                  handlers).
    set (masks := map (fun '(k, h) => (k, handler_mask sol false h)) handlers).
    set (CR := map (fun '(k, h) =>
                      (k, combine (handler_variables h)
                            (handler_mask sol false h)))
                 handlers).
    set (S := map (fun '(k, h) =>
                     (k, flat_map shape_of_param
                           (combine (handler_variables h)
                              (handler_mask sol false h))))
                handlers).
    set (C' := CR ++ C).
    assert (Hhs' : ∀ (k : continuation) (h : continuation_handler),
               (k, h) ∈ handlers ->
               local_handler unboxed sol (env_kinds (ue_env en)) C'
                 (handler_mask sol false h) h = true)
      by (intros k h Hin; exact (Hhs _ Hin)).
    clear Hhs.
    assert (HC' : conts_match C' (ue_decisions
                                   (ubind_continuations en dss))).
    { cbn [ue_decisions ubind_continuations]. apply Forall2_app; [ | exact HC ].
      exact (conts_match_rec _ C' handlers Hhs'). }
    assert (Hpen : pen_of (ubind_continuations en dss) C'
                   = bind_continuations (pen_of en C) masks).
    { unfold pen_of, C', bind_continuations.
      cbn [ue_env ubind_continuations env_conts env_kinds].
      rewrite map_app.
      match goal with
      | |- context [map ?f CR] => assert (Hm : map f CR = masks)
      end.
      { unfold CR, masks. rewrite map_map.
        apply map_ext_in. intros [k h] Hin. rewrite map_snd_combine;
          [ reflexivity | ].
        symmetry. exact (proj2 (handler_match _ _ false h (Hhs' k h Hin))). }
      f_equal; try reflexivity. rewrite <- Hm. reflexivity. }
    assert (HK : K_of C' = S ++ K_of C).
    { unfold C', K_of, CR, S. rewrite map_app, map_map. f_equal.
      apply map_ext. intros [k h]. reflexivity. }
    destruct (IH (ubind_continuations en dss) C' _
                (uinv_continuations en dss) HC' Hbody Hbody_edges)
      as (IHub & IHfc & IHbs).
    rewrite Hpen in IHub, IHfc, IHbs.
    assert (Hhandlers : ∀ hs : list (continuation * continuation_handler),
      (∀ (k : continuation) (h : continuation_handler),
         (k, h) ∈ hs -> (k, h) ∈ handlers) ->
      let hps :=
        map
          (fun '(k, h) =>
             (k,
              rebuild_handler F sol preserve_direct_calls unit_code_ids
                first_fresh (bind_continuations (pen_of en C) masks) false
                (handler_mask sol false h) h))
          hs
      in
      let hus :=
        map
          (fun '(k, h) =>
             (k,
              rebuild_handler_u false F sol cannot_change code_deps
                preserve_direct_calls unit_code_ids codes unboxed
                first_fresh_u [] (ubind_continuations en dss) false
                (handler_decisions false sol unboxed false h) h))
          hs
      in
      ubh_handlers unboxed (K_of C') hps hus
        (map (fun '(k, h) =>
                (k, flat_map shape_of_param
                      (combine (handler_variables h)
                         (handler_mask sol false h))))
           hs)
      ∧ handlers_bind_statics hps = handlers_bind_statics hus
      ∧ flat_map (fun '(_, h) => free_continuations_of_handler h) hps
        = flat_map (fun '(_, h) => free_continuations_of_handler h) hus
      ∧ map fst hps = map fst hus).
    { induction hs as [ | [k h] hs IHhs]; intros Hsub; cbv zeta;
        [ cbn [map]; split; [ apply ubh_handlers_nil | ];
          split; [ reflexivity | split; reflexivity ] | ].
      destruct IHhs as (Hub_hs & Hbs_hs & Hfc_hs & Hfst_hs).
      { intros k' h' Hin. apply Hsub. right. exact Hin. }
      assert (Hin : (k, h) ∈ handlers) by (apply Hsub; left; reflexivity).
      destruct (IHh k h Hin (ubind_continuations en dss) C' _ false
                  (uinv_continuations en dss) HC' (Hhs' _ _ Hin)
                  (every_in _ _ (k, h) Hhs_edges Hin) false)
        as (ps & ps' & md & md' & hp & hu & Hp & Hu & Hrel & Hmd & Hmd' & _
            & _ & Hhub & Hhfc & Hhbs).
      rewrite Hpen in Hp. cbv zeta. cbn [map]. rewrite Hp, Hu.
      split; [ apply ubh_handlers_cons; assumption | ].
      unfold handlers_bind_statics in *. cbn [existsb flat_map map fst].
      cbn [free_continuations_of_handler handler_binds_statics].
      rewrite Hhbs, Hbs_hs, Hhfc, Hfc_hs, Hfst_hs. split; [ | split ];
        reflexivity. }
    destruct (Hhandlers handlers (fun k h Hin => Hin))
      as (Hub_hs & Hbs_hs & Hfc_hs & Hfst_hs).
    fold S in Hub_hs.
    rewrite HK in Hub_hs, IHub.
    split; [ apply (ubh_let_cont_rec unboxed (K_of C) S); assumption | ].
    unfold handlers_bind_statics in Hbs_hs.
    cbn [free_continuations_of_expr binds_statics].
    rewrite Hfst_hs, Hfc_hs, IHfc, IHbs, Hbs_hs. split; reflexivity.
  - (* Apply *)
    intros ap en C ce Hen HC Hloc. discriminate Hloc.
  - (* Apply_cont *)
    intros ac en C ce Hen HC Hloc _. unfold plain_rebuild, unbox_rebuild.
    cbn [rebuild_expr rebuild_expr_u]. simpl in Hloc.
    unfold rebuild_apply_cont, rebuild_apply_cont_u.
    destruct (rewrite_apply_cont_rel en C ac HC Hloc)
      as [[Hp Hu] | (ac_p & ac_u & Hp & Hu & Hrel)];
      rewrite Hp, Hu; [ exact (ubh_result_invalid _) | ].
    split; [ constructor; exact Hrel | ].
    cbn [free_continuations_of_expr binds_statics].
    rewrite (ac_rel_conts _ _ _ Hrel). split; reflexivity.
  - (* Switch *)
    intros scrutinee arms en C ce Hen HC Hloc _. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hs Harms].
    unfold plain_rebuild, unbox_rebuild. cbn [rebuild_expr rebuild_expr_u].
    unfold rebuild_switch, rebuild_switch_u.
    assert (Hcompact : ∀ (d : Z) (ac : apply_cont),
               (d, ac) ∈ Keyed.compact Z.eqb arms ->
               apply_cont_ok unboxed sol (env_kinds (ue_env en)) C ac = true).
    { intros d ac Hin.
      apply (keyed_in_compact Z.eqb z_eqb_decides_equality) in Hin.
      apply (keyed_lookup_in Z.eqb z_eqb_decides_equality) in Hin.
      rewrite forallb_forall in Harms. exact (Harms _ Hin). }
    assert (Harms' :
      ∀ l : list (Z * apply_cont),
        (∀ (d : Z) (ac : apply_cont), (d, ac) ∈ l ->
           apply_cont_ok unboxed sol (env_kinds (ue_env en)) C ac = true) ->
        Forall2
          (fun arm arm' : Z * apply_cont =>
             fst arm' = fst arm ∧ ac_rel unboxed (K_of C) (snd arm) (snd arm'))
          (flat_map
             (fun '(d, ac) =>
                match rewrite_apply_cont sol (pen_of en C) ac with
                | Some ac => [(d, ac)]
                | None => []
                end) l)
          (flat_map
             (fun '(d, ac) =>
                match rewrite_apply_cont_u sol unboxed en ac with
                | Some ac => [(d, ac)]
                | None => []
                end) l)).
    { induction l as [ | [d ac] l IHl]; intros Hl; [ constructor | ].
      assert (Hrest : Forall2
          (fun arm arm' : Z * apply_cont =>
             fst arm' = fst arm ∧ ac_rel unboxed (K_of C) (snd arm) (snd arm'))
          (flat_map
             (fun '(d, ac) =>
                match rewrite_apply_cont sol (pen_of en C) ac with
                | Some ac => [(d, ac)]
                | None => []
                end) l)
          (flat_map
             (fun '(d, ac) =>
                match rewrite_apply_cont_u sol unboxed en ac with
                | Some ac => [(d, ac)]
                | None => []
                end) l)).
      { apply IHl. intros d' ac' Hin. apply (Hl d' ac'). right. exact Hin. }
      cbn [flat_map].
      destruct (rewrite_apply_cont_rel en C ac HC (Hl d ac (or_introl eq_refl)))
        as [[Hp Hu] | (ac_p & ac_u & Hp & Hu & Hrel)];
        rewrite Hp, Hu; [ exact Hrest | ].
      constructor; [ split; [ reflexivity | exact Hrel ] | exact Hrest ]. }
    pose proof (Harms' _ Hcompact) as Hrel.
    rewrite (rewrite_simple_kinds (pen_of en C) (ue_env en) scrutinee eq_refl).
    rewrite (rewrite_simple_u _ scrutinee Hs).
    destruct Hrel as [ | arm arm' l l' Harm Hrel ];
      [ exact (ubh_result_invalid _) | ].
    cbn [Keyed.is_empty].
    split; [ apply ubh_switch; [ apply rewrite_simple_plain, Hs | ] | ].
    { constructor; assumption. }
    split; [ | reflexivity ].
    cbn [free_continuations_of_expr flat_map].
    destruct Harm as [_ Hac]. rewrite (ac_rel_conts _ _ _ Hac). f_equal.
    clear -Hrel. induction Hrel as [ | a a' l l' [_ Ha] _ IH ];
      [ reflexivity | ].
    cbn [flat_map]. rewrite (ac_rel_conts _ _ _ Ha), IH. reflexivity.
  - (* Invalid *)
    intros en C ce Hen HC Hloc _. exact (ubh_result_invalid _).
  - (* Handler *)
    intros params metadata handler IH en C ce is_exn Hen HC Hloc Hedges dead.
    simpl in Hloc.
    apply andb_true_iff in Hloc as [Hloc Hbody].
    apply andb_true_iff in Hloc as [Hlen Hparams]. apply Nat.eqb_eq in Hlen.
    pose proof (with_kinds_combine params (hm_params_kinds metadata) Hlen)
      as Hbound.
    set (bound := with_kinds params (hm_params_kinds metadata)) in *.
    set (mask := handler_mask sol is_exn (Handler params metadata handler))
      in *.
    assert (Hmlen : length mask = length bound)
      by (unfold mask, handler_mask; apply params_to_keep_length).
    assert (Hfst : map fst bound = params)
      by (rewrite Hbound; apply map_fst_combine, Hlen).
    assert (Hsnd : map snd bound = hm_params_kinds metadata)
      by (rewrite Hbound; apply map_snd_combine, Hlen).
    rewrite <- Hbound in Hparams.
    destruct (params_rel_handler bound mask Hmlen Hparams) as [Hrel Hmatch].
    rewrite Hfst in Hrel, Hmatch.
    assert (Hbody' : local_expr unboxed sol
                       (env_kinds (ue_env (ubind_variables en bound))) C
                       handler = true).
    { cbn [ue_env ubind_variables ubind_variables_then ubind_set
           bind_variables_then env_kinds].
      rewrite Hbound. exact Hbody. }
    destruct (IH _ C ce (uinv_bind en bound) HC Hbody' Hedges)
      as (IHub & IHfc & IHbs).
    rewrite pen_bind in IHub, IHfc, IHbs.
    rewrite plain_rebuild_handler, unbox_rebuild_handler. cbv zeta.
    fold bound.
    exists (filter_kept mask bound),
      (params_of_decisions bound
         (handler_decisions false sol unboxed is_exn
            (Handler params metadata handler))),
      (with_params_kinds metadata
         (filter_kept mask (hm_params_kinds metadata))),
      (with_params_kinds metadata
         (map snd
            (params_of_decisions bound
               (handler_decisions false sol unboxed is_exn
                  (Handler params metadata handler))))).
    eexists _, _.
    split; [ rewrite <- filter_kept_map, Hfst; reflexivity | ].
    split; [ reflexivity | ].
    split; [ exact Hrel | ].
    split; [ rewrite hm_with_params_kinds, <- Hsnd, filter_kept_map;
             reflexivity | ].
    split; [ apply hm_with_params_kinds | ].
    split; [ exact Hmatch | ].
    split; [ rewrite Hmlen; unfold bound; apply with_kinds_length | ].
    destruct (dead && _); [ exact (ubh_result_invalid _) | ].
    split; [ exact IHub | split; assumption ].
Qed.


End Rebuilds_heap.

Theorem rebuild_unbox_refines_rebuild_heap :
  rebuild_unbox_refines_rebuild_heap_statement.
Proof.
  intros flag is_local u e T cannot_change code_deps dominated to_unbox
    preserve_direct_calls Hslots Hfragment Hclosed Hlocal Hcheck.
  set (F := traverse_unit flag u e) in *.
  unfold local_unboxing in Hlocal.
  pose proof (unit_unboxed_fields_fresh F T dominated to_unbox e)
    as [Hdistinct Hfresh].
  unfold rebuild_unit_unbox, rebuild_unit_unbox_repr,
    rebuild_unit_unbox_repr_mode.
  destruct (unit_unboxed_fields F T dominated to_unbox e) as [unboxed next]
    eqn:Hunboxed.
  cbn [fst] in Hlocal, Hdistinct, Hfresh.
  rename Hlocal into Hloc.
  pose proof (closure_free_reads_blocks flag u e
                (proj1 (local_closure_free unboxed (solution_of_tables T))
                   e [] [] Hloc))
    as Hreads.
  apply (proj1 (strong_local unboxed (solution_of_tables T)) e [] [] Hloc)
    in Hfresh.
  rename Hfresh into Hloc'.
  assert (Hchanged :
    unit_changed_representations (fun _ => false) F T dominated to_unbox [] e
    = []) by reflexivity.
  rewrite Hchanged.
  pose proof (check_closed_sound is_local F T Hclosed) as Hcl.
  destruct (FlambdaReaperFlowProofs.traverse_unit_facts_gen flag u e True true
              Hfragment (or_introl I))
    as (noted & deps & cds & Hedges & _).
  assert (Hfields_read :
    ∀ (x : variable) (fields : unboxed_fields) (f : field)
      (v : unboxed_field),
      fields_of unboxed x = Some fields -> (f, v) ∈ fields ->
      ∃ w base : node,
        g_accessor (graph_of_facts F) w f base
        ∧ s_usages (solution_of_tables T) (Node_variable x) base = true
        ∧ s_has_usage (solution_of_tables T) w = true).
  { intros x fields f v Hx Hin. apply (fields_of_in unboxed) in Hx.
    unfold unit_unboxed_fields in Hunboxed.
    assert (Hx' : (Node_variable x, fields)
                  ∈ fst (unboxed_fields_of_names F T (bound_code_my_closures e)
                           dominated to_unbox (fresh_name (names_of e))))
      by (rewrite Hunboxed; exact Hx).
    destruct (unboxed_fields_of_names_in F T _ dominated to_unbox _ _ _ Hx')
      as (a & next' & ->).
    destruct (mk_unboxed_fields_in F T _ dominated to_unbox _ a _ next' f _
                Hin)
      as (use & Huse).
    destruct (get_fields_in F T _ f use Huse)
      as (w & base & Hacc & Hbase & Hw).
    rewrite (add_usages_none F T _ Hreads) in Hbase.
    apply mem_node_In in Hbase.
    destruct (get_direct_usages_in T _ _ Hbase) as [Hus _].
    exists w, base. split; [ exact Hacc | split; [ exact Hus | ] ].
    destruct Hw as [Hw | Hw]; [ | exact Hw ].
    exact (has_usage_any _ _ _ Hcl _ Hw). }
  assert (Hreads_found :
    ∀ (x y : variable) (kind : flambda_kind) (f : field)
      (fields : unboxed_fields),
      fields_of unboxed x = None -> fields_of unboxed y = Some fields ->
      g_accessor (graph_of_facts F) (Node_variable x) f (Node_variable y) ->
      is_dead_var (solution_of_tables T) x kind = false ->
      is_var_used (solution_of_tables T) x kind = true ->
      kind = Value ∨ (∃ nk : naked_number_kind, kind = Naked_number nk) ->
      ∃ v : variable,
        Keyed.lookup field_eqb f fields = Some (Not_unboxed v)).
  { intros x y kind f fields Hx Hy Hacc Hdead Hused Hkind.
    assert (Hsx : mem_node (Node_variable x) (t_has_source T) = true).
    { unfold is_dead_var in Hdead.
      destruct Hkind as [-> | (nk & ->)]; simpl in Hdead;
        apply negb_false_iff in Hdead; exact Hdead. }
    assert (Hux : mem_node (Node_variable x) (t_has_usage T) = true).
    { unfold is_var_used in Hused.
      destruct Hkind as [-> | (nk & ->)]; exact Hused. }
    unfold unit_unboxed_fields in Hunboxed.
    assert (Hhx : has_to_be_unboxed dominated to_unbox (Node_variable x)
                  = false).
    { destruct (has_to_be_unboxed dominated to_unbox (Node_variable x))
        eqn:Hh; [ | reflexivity ].
      unfold has_to_be_unboxed in Hh.
      apply existsb_exists in Hh as ([a x'] & Hin & Htest).
      apply andb_true_iff in Htest as [Hx' Ha].
      apply node_eqb_eq in Hx'. subst x'.
      pose proof (unboxed_fields_of_names_bound F T
                    (bound_code_my_closures e) dominated to_unbox
                    (fresh_name (names_of e)) a (Node_variable x) Hin Ha)
        as Hb.
      rewrite Hunboxed in Hb. cbn [fst] in Hb. unfold fields_of in Hx.
      exfalso.
      exact (keyed_lookup_none_not_in node_eqb node_eqb_decides _ _ Hx Hb). }
    apply (fields_of_in unboxed) in Hy.
    assert (Hy' : (Node_variable y, fields)
                  ∈ fst (unboxed_fields_of_names F T
                           (bound_code_my_closures e) dominated to_unbox
                           (fresh_name (names_of e))))
      by (rewrite Hunboxed; exact Hy).
    destruct (unboxed_fields_of_names_dom F T _ dominated to_unbox _ _ _ Hy')
      as (a & next' & Hdom & Ha & ->).
    unfold check_unboxed_reads in Hcheck. rewrite forallb_forall in Hcheck.
    specialize (Hcheck _ Hacc). cbn beta iota in Hcheck.
    rewrite Hsx, Hux, Hhx in Hcheck. simpl in Hcheck.
    rewrite forallb_forall in Hcheck. specialize (Hcheck _ Hdom).
    cbn beta iota in Hcheck. rewrite node_eqb_refl, Ha in Hcheck.
    simpl in Hcheck. apply andb_true_iff in Hcheck as [Hnau Hsrc].
    apply negb_true_iff in Hnau.
    assert (Hblock : is_block_like_field f = true).
    { rewrite forallb_forall in Hreads. exact (Hreads _ Hacc). }
    assert (Hfs : is_function_slot f = false)
      by (destruct f; try discriminate Hblock; reflexivity).
    assert (Hdata : is_data_field f = true)
      by (destruct f; try discriminate Hblock; reflexivity).
    assert (Hus : mem_pair (Node_variable y) (Node_variable y) (t_usages T)
                  = true)
      by exact (usages_accessor _ _ _ Hcl _ _ _ Hacc Hnau).
    assert (Hhu : mem_node (Node_variable y) (t_has_usage T) = true)
      by exact (has_usage_accessor _ _ _ Hcl _ _ _ Hux Hacc).
    assert (Huses : mem_node (Node_variable y)
                      (add_usages_through_function_slots F T
                         (bound_code_my_closures e) true
                         (get_direct_usages T [Node_variable y]))
                    = true).
    { rewrite (add_usages_none F T _ Hreads). apply mem_node_In.
      apply get_direct_usages_self; assumption. }
    destruct (get_fields_found F T _ _ _ _ Hacc Huses Hfs Hux)
      as [Hex Hall].
    apply (mk_unboxed_fields_lookup F T _ dominated to_unbox _ a f Hdata).
    - intros Hno. rewrite Hno in Hsrc. discriminate Hsrc.
    - exact Hex.
    - intros use Hin.
      destruct (Hall use Hin) as [-> | (flow_to & -> & Hw)];
        [ left; reflexivity | right ].
      exists flow_to. split; [ reflexivity | ].
      destruct (forallb _ flow_to) eqn:Hft; [ | reflexivity ].
      rewrite forallb_forall in Hft. rewrite (Hft _ Hw) in Hhx.
      discriminate Hhx. }
  destruct (proj1 (rebuild_ubh F (solution_of_tables T)
                     (fun n => mem_node n cannot_change) code_deps
                     preserve_direct_calls (static_names code_ids_bound e)
                     (bound_code_metadata e) unboxed
                     (fresh_name (names_of e)) next noted deps cds
                     (unit_kept e) (unit_codes e)
                     (static_names symbols_of_binding e) is_local Hcl Hslots
                     Hfields_read Hreads_found)
              e empty_unbox_env [] (unit_conts u) uinv_empty (Forall2_nil _)
              Hloc' Hedges)
    as (Hub & _ & _).
  exact (ubh_refines unboxed Hdistinct _ _ Hub).
Qed.


(* Refinement up to the heap, then refinement, is refinement up to the
   heap. *)
Lemma refines_up_to_heap_refines (s m t : state) :
  refines_up_to_heap s m -> refines m t -> refines_up_to_heap s t.
Proof.
  intros (Hexit & Hback & Hwrong) (Hexit' & Hback').
  split; [ | split ].
  - intros k vs hp Hs. destruct (Hexit k vs hp Hs) as (vs' & hp' & Hm & Hvs).
    exists vs', hp'. split; [ exact (Hexit' k vs' hp' Hm) | exact Hvs ].
  - intros k vs' hp' Ht.
    destruct (Hback' (Exits k vs' hp') ltac:(discriminate) Ht) as [Hm | Hm].
    + exact (Hback k vs' hp' Hm).
    + right. exact (Hwrong Hm).
  - intros Ht. destruct (Hback' Goes_wrong ltac:(discriminate) Ht) as [Hm | Hm];
      exact (Hwrong Hm).
Qed.

Theorem rebuild_unbox_refines_up_to_heap_local :
  rebuild_unbox_refines_up_to_heap_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls Hlocal_slots Hfragment Hstatic Hkinded
    Hwell_formed Hone Hclosed Hlrs Hexact Hlocal Hcheck.
  apply (refines_up_to_heap_refines _
           (State (rebuild_unit_from_tables (traverse_unit flag u e) T
                     preserve_direct_calls e)
              [] [] empty_static_env empty_heap no_reads)).
  - exact (rebuild_refines_up_to_heap_local flag is_local u e T
             preserve_direct_calls Hlocal_slots Hfragment
             (static_constants_any_kind_arguments _ e Hstatic)
             Hkinded Hwell_formed Hone Hclosed Hlrs Hexact).
  - exact (rebuild_unbox_refines_rebuild_heap flag is_local u e T
             cannot_change (unit_code_deps flag u e) dominated to_unbox
             preserve_direct_calls Hlocal_slots Hfragment Hclosed Hlocal
             Hcheck).
Qed.

Theorem rebuild_unbox_refines_up_to_heap :
  rebuild_unbox_refines_up_to_heap_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls Hlocal_slots Hprojections Hfragment Hstatic Hkinded
    Hwell_formed Hone Hclosed Hexact Hlocal Hcheck.
  exact (rebuild_unbox_refines_up_to_heap_local flag is_local u e T
           cannot_change dominated to_unbox preserve_direct_calls Hlocal_slots
           Hfragment Hstatic Hkinded Hwell_formed Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hlocal_slots Hprojections)
           Hexact Hlocal Hcheck).
Qed.

Theorem heap_example : heap_example_statement.
Proof.
  split; [ intros f Hf; discriminate Hf | ].
  repeat (split; [ vm_compute; reflexivity | ]).
  vm_compute; reflexivity.
Qed.

Theorem cell_example : cell_example_statement.
Proof.
  do 9 (split; [ vm_compute; reflexivity | ]).
  assert (Hplain :
    behaves (State CellExample.plain [] [] empty_static_env empty_heap
               no_reads)
      (Exits 1%positive [CellExample.result] CellExample.heap_at_exit)).
  { exists 10, no_reads. vm_compute. reflexivity. }
  assert (Hrebuilt :
    behaves (State CellExample.rebuilt [] [] empty_static_env empty_heap
               no_reads)
      (Exits 1%positive [CellExample.result] empty_heap)).
  { exists 10, no_reads. vm_compute. reflexivity. }
  assert (Hobs : exits_observed_equal [CellExample.result]
                   CellExample.heap_at_exit [CellExample.result] empty_heap).
  { exists (fun _ _ => False).
    split; [ split; [ intros l l1 l2 [] | split; [ intros l1 l2 l' [] | ] ]
           | ].
    - intros l l' [].
    - cbn. repeat split. }
  split.
  { intros [Hexits _].
    pose proof (behaviour_unique _ _ _ (Hexits _ _ _ Hplain) Hrebuilt)
      as Heq.
    discriminate Heq. }
  split; [ | split ].
  - intros k vs hp Hs.
    pose proof (behaviour_unique _ _ _ Hplain Hs) as Heq.
    injection Heq as <- <- <-.
    exists [CellExample.result], empty_heap.
    split; [ exact Hrebuilt | exact Hobs ].
  - intros k vs' hp' Ht. left.
    pose proof (behaviour_unique _ _ _ Hrebuilt Ht) as Heq.
    injection Heq as <- <- <-.
    exists [CellExample.result], CellExample.heap_at_exit.
    split; [ exact Hplain | exact Hobs ].
  - intros Ht. pose proof (behaviour_unique _ _ _ Hrebuilt Ht) as Heq.
    discriminate Heq.
Qed.

