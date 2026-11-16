(* The lemmas of FlambdaContextProofs.v that are named [_where], at a
   property of the static environment, the heap and the read record
   together.

   FlambdaContextProofs.v relates corresponding handlers, and
   corresponding code, from the states that have a predicate that a step
   keeps ([inv]); its lemmas named [_where] are the case of a property
   of the static environment that every extension keeps. A pass that
   keeps a fact about a primitive that reads a cell needs a property
   that reads the heap and the read record as well: that the primitive,
   run again, gives the value that a variable holds and changes
   neither. Such a property is
   kept where the static environment is extended and where the heap and
   the record evolve (FlambdaSemanticsSpec.memory_evolves), and a step
   does only that (FlambdaSemanticsSpec.step_memory_evolves_statement).
   So the lemmas of that file hold at it. They are said here again, word
   for word, with the property applied to the three.

   [ok] may close over the variables of a definition, which are fixed.
   It does not read the variables of the state as they change: a pass
   still owes, at the entry of a handler, that what it knows of the
   variables of the definition holds with the parameters bound. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaContextSpec FlambdaContextProofs.
Import ListNotations.

(* A handler refines another where it runs: under the variables of its
   closure with its parameters bound, any continuations, and any static
   environment of which [ok] holds. *)
Definition handler_refines_where_memory
    (ok : static_env -> heap -> read_record -> Prop)
    (params : list variable) (kinds : list flambda_kind) (def_vars : env)
    (source target : expr) : Prop :=
  ∀ (vs : list value) (handler_vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    ok statics hp rd ->
    map kind_of_value vs = kinds ->
    bind_params params vs def_vars = Some handler_vars ->
    refines (State source handler_vars conts statics hp rd)
      (State target handler_vars conts statics hp rd).

Definition expr_refines_where_memory
    (ok : static_env -> heap -> read_record -> Prop)
    (source target : expr) : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    ok statics hp rd ->
    refines (State source vars conts statics hp rd)
      (State target vars conts statics hp rd).

(* The body of code refines another where it runs: at a call, under its
   parameters bound to values of the kinds of its arity and [my_closure] to
   a value of kind [Value], any continuations, and any static environment
   of which [ok] holds. The kinds are those of the values; nothing is
   asked of the kinds that a term gives. *)
Definition code_refines_where_memory
    (ok : static_env -> heap -> read_record -> Prop)
    (params : list variable) (my_closure : variable)
    (metadata : code_metadata) (source target : expr) : Prop :=
  ∀ (vs : list value) (callee : value) (callee_vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record),
    ok statics hp rd ->
    map kind_of_value vs = unarize (cm_params_arity metadata) ->
    kind_of_value callee = Value ->
    bind_params params vs [(my_closure, callee)] = Some callee_vars ->
    refines (State source callee_vars conts statics hp rd)
      (State target callee_vars conts statics hp rd).

Section Where_memory.

Variable ok : static_env -> heap -> read_record -> Prop.

Hypothesis ok_extend :
  ∀ (statics statics' : static_env) (hp : heap) (rd : read_record),
    ok statics hp rd -> statics_extend statics statics' -> ok statics' hp rd.

Hypothesis ok_evolves :
  ∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record),
    ok statics hp rd -> memory_evolves hp rd hp' rd' -> ok statics hp' rd'.

(* [ok] of the static environment, the heap and the record of a state:
   a predicate on states that a step keeps, since a step extends the
   first and makes the other two evolve, and that does not look at the
   expression. The lemmas of this section are those of
   FlambdaContextProofs.v at this predicate. *)
Definition memory_ok (s : state) : Prop :=
  ok (state_statics s) (state_heap s) (state_reads s).

Lemma memory_ok_step (s s' : state) :
  memory_ok s -> step s = Step s' -> memory_ok s'.
Proof.
  intros Hok Hstep. unfold memory_ok.
  exact (ok_evolves _ _ _ _ _
           (ok_extend _ _ _ _ Hok (step_extends_statics s s' Hstep))
           (step_memory_evolves s s' Hstep)).
Qed.

Lemma memory_ok_refines_from (source target : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  (ok statics hp rd ->
   refines (State source vars conts statics hp rd)
     (State target vars conts statics hp rd)) ->
  refines_from memory_ok (State source vars conts statics hp rd)
    (State target vars conts statics hp rd).
Proof.
  intros Hrefines. split; [ split; intros Hok; exact Hok | exact Hrefines ].
Qed.

Lemma handler_refines_where_memory_from (params : list variable)
    (kinds : list flambda_kind) (def_vars : env) (source target : expr)
    (vs : list value) (handler_vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  handler_refines_where_memory ok params kinds def_vars source target ->
  map kind_of_value vs = kinds ->
  bind_params params vs def_vars = Some handler_vars ->
  refines_from memory_ok (State source handler_vars conts statics hp rd)
    (State target handler_vars conts statics hp rd).
Proof.
  intros Hrefines Hkinds Hbind. apply memory_ok_refines_from. intros Hok.
  exact (Hrefines vs handler_vars conts statics hp rd_ Hok Hkinds Hbind).
Qed.

Lemma expr_refines_where_memory_from (source target : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  expr_refines_where_memory ok source target ->
  refines_from memory_ok (State source vars conts statics hp rd)
    (State target vars conts statics hp rd).
Proof.
  intros Hrefines. apply memory_ok_refines_from. intros Hok.
  exact (Hrefines vars conts statics hp rd_ Hok).
Qed.

Lemma code_refines_where_memory_from (params : list variable)
    (my_closure : variable) (metadata : code_metadata)
    (source target : expr) (vs : list value) (callee : value)
    (callee_vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  code_refines_where_memory ok params my_closure metadata source target ->
  bind_params params vs [(my_closure, callee)] = Some callee_vars ->
  map kind_of_value vs = unarize (cm_params_arity metadata) ->
  kind_of_value callee = Value ->
  refines_from memory_ok (State source callee_vars conts statics hp rd)
    (State target callee_vars conts statics hp rd).
Proof.
  intros Hrefines Hbind Hkinds Hvalue. apply memory_ok_refines_from.
  intros Hok.
  exact
    (Hrefines vs callee callee_vars conts statics hp rd_ Hok Hkinds Hvalue
       Hbind).
Qed.

(* An expression under environments refines it under ones whose closures
   have refining handlers and whose code has refining bodies. The
   relations are needed in both directions, as the two halves of [refines]
   run different sides. *)
Lemma related_states_refines_memory (e : expr) (vars : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp : heap)
      (rd : read_record) :
  ok statics hp rd -> ok statics' hp rd ->
  cont_env_related (handler_refines_where_memory ok) conts conts' ->
  cont_env_related
    (fun params kinds def_vars target source =>
       handler_refines_where_memory ok params kinds def_vars source target)
    conts' conts ->
  statics_related (expr_refines_where_memory ok) statics statics' ->
  statics_related
    (fun target source => expr_refines_where_memory ok source target)
    statics' statics ->
  refines (State e vars conts statics hp rd)
    (State e vars conts' statics' hp rd).
Proof.
  intros Hok Hok' Hrel Hrel' Hstatics Hstatics'.
  exact
    (related_states_refines_from memory_ok memory_ok_step
       (handler_refines_where_memory ok)
       (fun params kinds def_vars target source =>
          handler_refines_where_memory ok params kinds def_vars source target)
       (ignoring_binders (expr_refines_where_memory ok))
       (ignoring_binders
          (fun target source => expr_refines_where_memory ok source target))
       handler_refines_where_memory_from
       (fun params kinds def_vars target source =>
          handler_refines_where_memory_from params kinds def_vars source target)
       (fun id params my_closure ret exn metadata source target vs callee
          callee_vars return_to exn_to statics'' hp'' rd'' slots Hrefines _ _ _
          _ =>
          expr_refines_where_memory_from source target callee_vars
            [(ret, return_to); (exn, exn_to)] statics'' hp'' rd'' Hrefines)
       (fun id params my_closure ret exn metadata target source vs callee
          callee_vars return_to exn_to statics'' hp'' rd'' slots Hrefines _ _ _
          _ =>
          expr_refines_where_memory_from source target callee_vars
            [(ret, return_to); (exn, exn_to)] statics'' hp'' rd'' Hrefines)
       e vars conts conts' statics statics' _ rd_ Hok Hok' Hrel Hrel' Hstatics
       Hstatics').
Qed.

(* The case of one static environment. *)
Lemma related_envs_refines_memory (e : expr) (vars : env)
    (conts conts' : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  ok statics hp rd ->
  cont_env_related (handler_refines_where_memory ok) conts conts' ->
  cont_env_related
    (fun params kinds def_vars target source =>
       handler_refines_where_memory ok params kinds def_vars source target)
    conts' conts ->
  refines (State e vars conts statics hp rd)
    (State e vars conts' statics hp rd).
Proof.
  intros Hok Hrel Hrel'.
  apply related_states_refines_memory; try assumption; apply
    statics_related_refl.
Qed.

(* A [Let_cont] refines the same with a handler replaced by one that
   refines it where it runs. *)
Lemma let_cont_handler_refines_where_memory (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' body : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  ok statics hp rd ->
  handler_refines_where_memory ok params (hm_params_kinds metadata) vars handler
    handler' ->
  refines
    (State
       (Let_cont_non_recursive k (Handler params metadata handler) body)
       vars conts statics hp rd)
    (State
       (Let_cont_non_recursive k (Handler params metadata handler') body)
       vars conts statics hp rd).
Proof.
  intros Hok Hhandler. apply let_cont_non_recursive_refines_step.
  apply related_envs_refines_memory; [ exact Hok | | ];
    (apply Forall2_cons; [ | apply cont_env_related_refl ]);
    (split; [ reflexivity | ]);
    (apply Closure_related;
      [ apply Handler_related; exact Hhandler | apply cont_env_related_refl
      | apply Forall2_nil ]).
Qed.

(* The same for recursive handlers, any number of them replaced. *)
Lemma let_cont_recursive_handlers_refine_where_memory
    (handlers handlers' : list (continuation * continuation_handler))
    (body : expr) (vars : env) (conts : cont_env) (statics : static_env)
      (hp : heap) (rd : read_record) :
  ok statics hp rd ->
  handlers_bind_statics handlers = handlers_bind_statics handlers' ->
  handlers_related (handler_refines_where_memory ok) vars handlers handlers' ->
  handlers_related
    (fun params kinds def_vars target source =>
       handler_refines_where_memory ok params kinds def_vars source target)
    vars handlers' handlers ->
  refines (State (Let_cont_recursive handlers body) vars conts statics hp rd)
    (State (Let_cont_recursive handlers' body) vars conts statics hp rd).
Proof.
  intros Hok Hbinds Hhandlers Hhandlers'.
  apply let_cont_recursive_refines_step; [ exact Hbinds | intros _ ].
  apply related_envs_refines_memory; [ exact Hok | | ];
    (apply Forall2_app; [ | apply cont_env_related_refl ]);
    (apply rec_closures_related; [ assumption | apply cont_env_related_refl ]).
Qed.

(* A [Let_static] refines the same with the body of a piece of code
   replaced by one that refines it where it runs. *)
Lemma let_static_code_refines_where_memory (before : list static_binding)
    (id : code_id) (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata)
    (code_body code_body' : expr) (after : list static_binding) (body : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  ok statics hp rd ->
  binds_statics code_body = binds_statics code_body' ->
  code_refines_where_memory ok params my_closure metadata code_body code_body'
    ->
  refines
    (State
       (Let_static
          (before
           ++ Bind_code id
                (Code params my_closure ret exn metadata code_body)
           :: after)
          body)
       vars conts statics hp rd)
    (State
       (Let_static
          (before
           ++ Bind_code id
                (Code params my_closure ret exn metadata code_body')
           :: after)
          body)
       vars conts statics hp rd).
Proof.
  intros Hok Hbinds Hcode.
  exact
    (let_static_code_refines_from memory_ok memory_ok_step
       (handler_refines_where_memory ok)
       (fun params kinds def_vars target source =>
          handler_refines_where_memory ok params kinds def_vars source target)
       (fun _ params my_closure _ _ metadata =>
          code_refines_where_memory ok params my_closure metadata)
       (fun _ params my_closure _ _ metadata target source =>
          code_refines_where_memory ok params my_closure metadata source target)
       handler_refines_where_memory_from
       (fun params kinds def_vars target source =>
          handler_refines_where_memory_from params kinds def_vars source target)
       (fun _ params my_closure ret exn metadata source target vs callee
          callee_vars return_to exn_to statics' hp' rd' _ Hrefines Hbind Hkinds
          Hvalue _ =>
          code_refines_where_memory_from params my_closure metadata source
            target
            vs callee callee_vars [(ret, return_to); (exn, exn_to)]
            statics' hp' rd' Hrefines Hbind Hkinds Hvalue)
       (fun _ params my_closure ret exn metadata target source vs callee
          callee_vars return_to exn_to statics' hp' rd' _ Hrefines Hbind Hkinds
          Hvalue _ =>
          code_refines_where_memory_from params my_closure metadata source
            target
            vs callee callee_vars [(ret, return_to); (exn, exn_to)]
            statics' hp' rd' Hrefines Hbind Hkinds Hvalue)
       before id params my_closure ret exn metadata code_body code_body'
       after body vars conts statics hp rd Hok Hok Hbinds Hcode Hcode).
Qed.

End Where_memory.
